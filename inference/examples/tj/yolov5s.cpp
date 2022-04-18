//
// Created by hsyuan on 2021-02-22.
//

#include "yolov5s.h"

#include <pthread.h>

#include "hiredis/hiredis.h"

int MAX_REDIS_LEN = 1024 * 1024 * 10;
int MAX_JSON_LEN = 1024 * 10;
extern redisContext *global_redis;
extern pthread_mutex_t global_redis_mutx;

YoloV5::YoloV5(bm::BMNNContextPtr bmctx, int max_batch) :
    m_bmctx(bmctx), MAX_BATCH(max_batch) {
    // the bmodel has only one yolo network.
    auto net_name = m_bmctx->network_name(0);
    m_bmnet = std::make_shared<bm::BMNNNetwork>(m_bmctx->bmrt(), net_name);
    assert(m_bmnet != nullptr);
    assert(m_bmnet->inputTensorNum() == 1);
    auto tensor = m_bmnet->inputTensor(0);

    // YOLOV5 input is NCHW
    m_net_h = tensor->get_shape()->dims[2];
    m_net_w = tensor->get_shape()->dims[3];
}

YoloV5::~YoloV5() {
}

void YoloV5::setLastDetector(bool isLast) {
    m_isLastDetector = isLast;
}

// int YoloV5::preprocess(std::vector<bm::FrameBaseInfo>& frames,
// std::vector<bm::FrameInfo>& frame_infos)
int YoloV5::preprocess(std::vector<bm::FrameInfo2> &frame_infos) {
    int ret = 0;
    bm_handle_t handle = m_bmctx->handle();
    std::cout << "YoloV5::preprocess called" << std::endl;
    std::cout << this->detectorName << std::endl;

    for (int frameInfoIdx = 0; frameInfoIdx < frame_infos.size();
         ++frameInfoIdx) {
        auto &frame_info = frame_infos[frameInfoIdx];

        int frameNum = frame_info.frames.size();
        assert(frame_info.forwards.size() == 0);

        // Check input
        int total = frame_info.frames.size();
        int left = (total % MAX_BATCH == 0 ? MAX_BATCH : total % MAX_BATCH);
        int batch_num =
            total % MAX_BATCH == 0 ? total / MAX_BATCH : (total / MAX_BATCH + 1);
        for (int batch_idx = 0; batch_idx < batch_num; ++batch_idx) {
            int num = MAX_BATCH;
            int start_idx = batch_idx * MAX_BATCH;
            if (batch_idx == batch_num - 1) {
                // last one
                num = left;
            }

            // std::cout << "yolov5 MAX_BATCH == " << MAX_BATCH << std::endl;
            // std::cout << "yolov5 num == " << num << std::endl;

            // bm::FrameInfo2 finfo;
            // 1. Resize
            bm_image resized_imgs[MAX_BATCH];
            ret = bm::BMImage::create_batch(handle, m_net_h, m_net_w,
                                            FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE,
                                            resized_imgs, num, 64);
            assert(BM_SUCCESS == ret);

            for (int i = 0; i < num; ++i) {
                bm_image image1;
                bm::BMImage::from_avframe(
                    handle, frame_info.frames[start_idx + i].avframe, image1, true);
                ret = bmcv_image_vpp_convert(handle, 1, image1, &resized_imgs[i]);
                assert(BM_SUCCESS == ret);

                // convert data to jpeg
                uint8_t *jpeg_data = NULL;
                size_t out_size = 0;
#if USE_QTGUI
                bmcv_image_jpeg_enc(handle, 1, &image1, (void **)&jpeg_data, &out_size);
#endif
                frame_info.frames[start_idx + i].jpeg_data =
                    std::make_shared<bm::Data>(jpeg_data, out_size);
                frame_info.frames[start_idx + i].height = image1.height;
                frame_info.frames[start_idx + i].width = image1.width;
                // av_frame_unref(frame_info.frames[start_idx + i].avframe);
                // av_frame_free(&frame_info.frames[start_idx + i].avframe);

                // finfo.frames.push_back(frame_info.frames[start_idx + i]);
                bm_image_destroy(image1);
            }

            // 2. Convert to
            bm_image convertto_imgs[MAX_BATCH];
            float alpha, beta;

            bm_image_data_format_ext img_type = DATA_TYPE_EXT_FLOAT32;
            auto inputTensorPtr = m_bmnet->inputTensor(0);
            if (inputTensorPtr->get_dtype() == BM_INT8) {
                img_type = DATA_TYPE_EXT_1N_BYTE_SIGNED;
                alpha = 0.847682119;
                beta = -0.5;
                img_type = (DATA_TYPE_EXT_1N_BYTE_SIGNED);
            } else {
                alpha = 1.0 / 255;
                beta = 0.0;
                img_type = DATA_TYPE_EXT_FLOAT32;
            }

            ret = bm::BMImage::create_batch(handle, m_net_h, m_net_w,
                                            FORMAT_RGB_PLANAR, img_type,
                                            convertto_imgs, num, 1, false, true);
            assert(BM_SUCCESS == ret);

            bm_tensor_t input_tensor = *inputTensorPtr->bm_tensor();
            bm::bm_tensor_reshape_NCHW(handle, &input_tensor, num, 3, m_net_h,
                                       m_net_w);

            ret = bm_image_attach_contiguous_mem(num, convertto_imgs,
                                                 input_tensor.device_mem);
            assert(BM_SUCCESS == ret);

            bmcv_convert_to_attr convert_to_attr;
            convert_to_attr.alpha_0 = alpha;
            convert_to_attr.alpha_1 = alpha;
            convert_to_attr.alpha_2 = alpha;
            convert_to_attr.beta_0 = beta;
            convert_to_attr.beta_1 = beta;
            convert_to_attr.beta_2 = beta;

            ret = bmcv_image_convert_to(m_bmctx->handle(), num, convert_to_attr,
                                        resized_imgs, convertto_imgs);
            assert(ret == 0);

            bm_image_dettach_contiguous_mem(num, convertto_imgs);

            bm::NetForward io;
            io.batch_size = num;
            io.input_tensors.push_back(input_tensor);
            for (int l = 0; l < m_bmnet->outputTensorNum(); ++l) {
                bm_tensor_t t;
                io.output_tensors.push_back(t);
            }

            frame_info.forwards.push_back(io);
            bm::BMImage::destroy_batch(resized_imgs, num);
            bm::BMImage::destroy_batch(convertto_imgs, num);

            // frame_infos.push_back(finfo);
        }
    }
}

int YoloV5::forward(std::vector<bm::FrameInfo2> &frame_infos) {
    int ret = 0;
    for (int b = 0; b < frame_infos.size(); ++b) {
        for (auto &fwd : frame_infos[b].forwards) {
            // for (int i = 0; i < m_bmnet->outputTensorNum(); ++i) {
            //     bm_tensor_t tensor;
            //     fwd.output_tensors.push_back(tensor);
            // }

#if DUMP_FILE
            bm::BMImage::dump_dev_memory(bmctx_->handle(),
                                         frame_infos[b].input_tensors[0].device_mem,
                                         "convertto", frame_infos[b].frames.size(),
                                         m_net_h, m_net_w, false, false);
#endif
            ret = m_bmnet->forward(fwd.input_tensors.data(), fwd.input_tensors.size(),
                                   fwd.output_tensors.data(),
                                   fwd.output_tensors.size());
            assert(BM_SUCCESS == ret);
        }
    }

    return 0;
}

void YoloV5::free_fwds(std::vector<bm::NetForward> &NetIOs) {
    for (auto &ios : NetIOs) {
        for (auto &tensor : ios.input_tensors) {
            bm_free_device(m_bmctx->handle(), tensor.device_mem);
        }

        for (auto &tensor : ios.output_tensors) {
            bm_free_device(m_bmctx->handle(), tensor.device_mem);
        }
    }
}

int YoloV5::postprocess(std::vector<bm::FrameInfo2> &frame_infos) {
    char *redis_buf = (char *)malloc(MAX_REDIS_LEN);
    char *json = (char *)malloc(MAX_JSON_LEN);
    char *listJson = (char *)malloc(MAX_JSON_LEN);
    for (int i = 0; i < frame_infos.size(); ++i) {
        memset(redis_buf, 0, MAX_REDIS_LEN);
        memset(json, 0, MAX_JSON_LEN);
        memset(listJson, 0, MAX_JSON_LEN);
        sprintf(listJson, "{\"StreamID\":\"%s\",\"list\":[",
                this->channelId.data());

        // Free AVFrames
        auto frame_info = frame_infos[i];

        // extract face detection
        extract_yolobox_cpu(frame_info);
        for (int i = 0; i < frame_info.out_datums.size(); i++) {
            bm::NetOutputDatum out = frame_info.out_datums[i];

            for (int i = 0; i < out.obj_rects.size(); i++) {
                bm::NetOutputObject obj = out.obj_rects[i];
                std::cout << this->detectorName << "width:" << obj.width();
                std::cout << " height:" << obj.height();
                std::cout << " score:" << obj.score;
                std::cout << " class_id:" << obj.class_id << std::endl;
                std::string labelName = get_label(obj.class_id);
                sprintf(listJson,
                        "%s{\"x1\":%f,\"y1\":%f,\"x2\":%f,\"y2\":%f,\"class\":\"%s\", "
                        "\"confidence\":%f,\"track_id\":%d}",
                        listJson, obj.x1, obj.y1, obj.x2, obj.y2, labelName.data(),
                        obj.score, 0);
                if (i != out.obj_rects.size() - 1) {
                    sprintf(listJson, "%s,", listJson);
                }
            }
            sprintf(listJson, "%s]}", listJson);
        }

        for (int i = 0; i < frame_info.frames.size(); i++) {
            bm::FrameBaseInfo2 frame = frame_info.frames[i];
            struct timeval tv;
            gettimeofday(&tv, NULL);
            int64_t curentTimeMs = tv.tv_sec * 1000 + tv.tv_usec / 1000;

            std::cout << "frame.avframe " << i << std::endl;
            // std::cout << "frame.avframe format " << frame.avframe->format <<
            // std::endl; std::cout << "frame.avframe format " << AV_PIX_FMT_NV12 <<
            // std::endl;
            int32_t yuvSize = 0;
            int32_t ySize = 0;
            int32_t uSize = 0;
            int32_t vSize = 0;
            int32_t uvSize = 0;
            int height = frame.avframe->height;
            int width = frame.avframe->width;
            // width = 1;
            if (frame.avframe->format == AV_PIX_FMT_NV12) {
                ySize = frame.avframe->linesize[0];
                uvSize = frame.avframe->linesize[1];
                vSize = frame.avframe->linesize[2];
                // uvSize = uvSize / 2;
                yuvSize = ySize * height + uvSize * height / 2;
                sprintf(json,
                        "{\"Long_j\":%d,\"width\":%d,\"height\":%d,\"timestamp\":%ld,"
                        "\"json\":%s,\"long_p\":%d}",
                        strlen(listJson), frame.avframe->width, frame.avframe->height,
                        curentTimeMs, listJson, yuvSize);

                printf("json %s\n", json);
                int jsonLen = strlen(json);
                int sizeInt = sizeof(int);

                int *redis_buf_int = (int *)redis_buf;
                *redis_buf_int = jsonLen;
                memcpy(redis_buf + sizeInt, json, jsonLen);

                printf("redis jsonLen, sizeInt, yuvSize %d %d %d\n", jsonLen, sizeInt,
                       yuvSize);
                printf("ySize %d, uvSize %d vsize %d\n", ySize, uvSize, vSize);
                // copy y data
                memcpy(redis_buf + sizeInt + jsonLen, frame.avframe->data[0],
                       ySize * height);
                // copy uv data
                memcpy(redis_buf + sizeInt + jsonLen + ySize * height,
                       frame.avframe->data[1], uvSize * height / 2);

                pthread_mutex_lock(&global_redis_mutx);
                redisReply *reply = (redisReply *)redisCommand(
                    global_redis, "PUBLISH %s %b ", redisTopic.data(), redis_buf,
                    sizeInt + jsonLen + yuvSize);
                printf("redis send %d %s %d\n", reply->type, reply->str,
                       reply->integer);
                freeReplyObject(reply);
                pthread_mutex_unlock(&global_redis_mutx);
                // char name[100] = "";
                // sprintf(name, "%ld_bgr.png", curentTimeMs);
                // FILE *fd = fopen(name, "wb");

                // if (fd == NULL) {
                //     perror("open failed!");
                //     exit(1);
                // }

                // fwrite(redis_buf + sizeInt + jsonLen, yuvSize, 1, fd);
                // fclose(fd);

                // cv::Mat BGR;
                // cv::Mat NV12 = cv::Mat(height * 3 / 2, width, CV_8UC1, redis_buf +
                // sizeInt + jsonLen);

                // cv::cvtColor(NV12, BGR, 91);
                // cv::imwrite(name, BGR);

                // std::cout << "this->detectorName redis topic " << redisTopic <<
                // std::endl; std::cout << "this->detectorName redis global_redis " <<
                // &global_redis << std::endl; redisReply *reply = (redisReply
                // *)redisCommand(global_redis, "PUBLISH %s %s ", redisTopic, redis_buf,
                // sizeInt + jsonLen); redisReply *reply = (redisReply
                // *)redisCommand(global_redis, "PUBLISH %s %s", "mychannel", "h3333
                // world", 3);

                // redisReply *reply = (redisReply *)redisCommand(global_redis, "PUBLISH
                // %s %s ", redisTopic.data(), "hellllllllllll", 10);
            }

            break;
        }

        // std::cout << "this->detectorName" << std::endl;
        // std::cout << this->detectorName << std::endl;
        // std::cout << "free_fwds" << std::endl;
        // free input and output tensors
        free_fwds(frame_info.forwards);
        frame_info.forwards.clear();

        // std::cout << "m_pfnDetectFinish" << std::endl;
        if (m_pfnDetectFinish != nullptr) {
            m_pfnDetectFinish(frame_info);
        }

        // std::cout << "m_nextInferPipe" << std::endl;
        if (m_nextInferPipe) {
            // std::cout << "m_nextInferPipe push_frame" << std::endl;
            m_nextInferPipe->push_frame(&frame_info);
        }

        // std::cout << "m_nextInferPipe 111" << std::endl;
        if (m_isLastDetector) {
            // std::cout << "m_isLastDetector this->detectorName" << std::endl;
            // std::cout << "=============================================" <<
            // std::endl; std::cout << this->detectorName << std::endl;
            for (int j = 0; j < frame_info.frames.size(); ++j) {
                auto reff = frame_info.frames[j];
                assert(reff.avpkt != nullptr);
                av_packet_unref(reff.avpkt);
                av_packet_free(&reff.avpkt);

                // assert(reff.avframe == nullptr);
                av_frame_unref(reff.avframe);
                av_frame_free(&reff.avframe);
            }
        }

        // // Free Tensors
        // for (auto &tensor : frame_info.input_tensors) {
        //     bm_free_device(m_bmctx->handle(), tensor.device_mem);
        // }

        // for (auto &tensor : frame_info.output_tensors) {
        //     bm_free_device(m_bmctx->handle(), tensor.device_mem);
        // }
    }
    free(json);
    free(listJson);
    free(redis_buf);
}

float YoloV5::sigmoid(float x) {
    return 1.0 / (1 + expf(-x));
}

int YoloV5::argmax(float *data, int num) {
    float max_value = 0.0;
    int max_index = 0;
    for (int i = 0; i < num; ++i) {
        float sigmoid_value = sigmoid(data[i]);
        if (sigmoid_value > max_value) {
            max_value = sigmoid_value;
            max_index = i;
        }
    }

    return max_index;
}

float YoloV5::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w,
                                      int dst_h, bool *pIsAligWidth) {
    float ratio;
    ratio = (float)dst_w / src_w;
    int dst_h1 = src_h * ratio;
    if (dst_h1 > dst_h) {
        *pIsAligWidth = false;
        ratio = (float)src_w / src_h;
    } else {
        *pIsAligWidth = true;
        ratio = (float)src_h / src_w;
    }

    return ratio;
}

void YoloV5::NMS(bm::NetOutputObjects &dets, float nmsConfidence) {
    int length = dets.size();
    int index = length - 1;

    std::sort(dets.begin(), dets.end(),
              [](const bm::NetOutputObject &a, const bm::NetOutputObject &b) {
                  return a.score < b.score;
              });

    std::vector<float> areas(length);
    for (int i = 0; i < length; i++) {
        areas[i] = dets[i].width() * dets[i].height();
    }

    while (index > 0) {
        int i = 0;
        while (i < index) {
            float left = std::max(dets[index].x1, dets[i].x1);
            float top = std::max(dets[index].y1, dets[i].y1);
            float right = std::min(dets[index].x1 + dets[index].width(),
                                   dets[i].x1 + dets[i].width());
            float bottom = std::min(dets[index].y1 + dets[index].height(),
                                    dets[i].y1 + dets[i].height());
            float overlap =
                std::max(0.0f, right - left) * std::max(0.0f, bottom - top);
            if (overlap / (areas[index] + areas[i] - overlap) > nmsConfidence) {
                areas.erase(areas.begin() + i);
                dets.erase(dets.begin() + i);
                index--;
            } else {
                i++;
            }
        }
        index--;
    }
}

void YoloV5::extract_yolobox_cpu(bm::FrameInfo2 &frameInfo) {
    std::vector<bm::NetOutputObject> yolobox_vec;
    std::vector<cv::Rect> bbox_vec;
    auto &images = frameInfo.frames;
    for (int batch_idx = 0; batch_idx < (int)images.size(); ++batch_idx) {
        yolobox_vec.clear();
        auto &frame = images[batch_idx];
        int frame_width = frame.width;
        int frame_height = frame.height;

#if USE_ASPECT_RATIO
        bool isAlignWidth = false;
        get_aspect_scaled_ratio(frame.cols, frame.rows, m_net_w, m_net_h,
                                &isAlignWidth);

        if (isAlignWidth) {
            frame_height = frame_width * (float)m_net_h / m_net_w;
        } else {
            frame_width = frame_height * (float)m_net_w / m_net_h;
        }
#endif

        int output_num = m_bmnet->outputTensorNum();
        int nout = m_class_num + 5;

        for (int fwd_idx = 0; fwd_idx < frameInfo.forwards.size(); ++fwd_idx) {
            for (int tidx = 0; tidx < output_num; ++tidx) {
                bm::BMNNTensor output_tensor(
                    m_bmctx->handle(), "", 1.0,
                    &frameInfo.forwards[fwd_idx].output_tensors[tidx]);
                int feat_h = output_tensor.get_shape()->dims[2];
                int feat_w = output_tensor.get_shape()->dims[3];
                int area = feat_h * feat_w;
                float *output_data =
                    (float *)output_tensor.get_cpu_data() + batch_idx * 3 * area * nout;
                for (int anchor_idx = 0; anchor_idx < m_anchor_num; anchor_idx++) {
                    int feature_size = feat_h * feat_w * nout;
                    float *ptr = output_data + anchor_idx * feature_size;
                    for (int i = 0; i < area; i++) {
                        float score = sigmoid(ptr[4]);
                        if (score > m_objThreshold) {
                            float centerX = (sigmoid(ptr[0]) * 2 - 0.5 + i % feat_w) * frame_width / feat_w;
                            float centerY = (sigmoid(ptr[1]) * 2 - 0.5 + i / feat_h) * frame_height / feat_h;                       // center_y
                            float width = pow((sigmoid(ptr[2]) * 2), 2) * m_anchors[tidx][anchor_idx][0] * frame_width / m_net_w;   // w
                            float height = pow((sigmoid(ptr[3]) * 2), 2) * m_anchors[tidx][anchor_idx][1] * frame_height / m_net_h; // h
                            bm::NetOutputObject box;
                            box.x1 = int(centerX - width / 2);
                            box.y1 = int(centerY - height / 2);
                            box.x2 = box.x1 + width;
                            box.y2 = box.y1 + height;
                            int class_id = argmax(&ptr[5], m_class_num);
                            box.score = sigmoid(ptr[class_id + 5]) * score;
                            box.class_id = class_id;

                            if (box.score >= m_confThreshold) {
                                yolobox_vec.push_back(box);
                            }
                        }
                        ptr += (m_class_num + 5);
                    }
                }
            } // end of tidx
        }

        NMS(yolobox_vec, m_nmsThreshold);
        bm::NetOutputDatum datum(yolobox_vec);
        frameInfo.out_datums.push_back(datum);
    }
}
