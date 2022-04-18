//
// Created by yuan on 3/4/21.
//

#include "opencv2/opencv.hpp"
#include "face_worker.h"
#include "configuration.h"
#include "bmutility_timer.h"
#include <iomanip>
#include <map>
#include "hiredis/hiredis.h"
#include <pthread.h>

redisContext *global_redis = nullptr;
pthread_mutex_t global_redis_mutx;

int main(int argc, char *argv[]) {
    const char *keys = "{help | 0 | Print help info}"
                       "{bmodel | /data/face_demo/models/face_demo.bmodel | input bmodel path}"
                       "{max_batch | 4 | Max batch size}"
                       "{output | None | Output stream URL}"
                       "{num | 1 | Channels to run}"
                       "{config | ./cameras.json | path to cameras.json}";

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }
    global_redis = redisConnect("127.0.0.1", 6379);
    if (global_redis == NULL || global_redis->err) {
        if (global_redis) {
            printf("Error: %s\n", global_redis->errstr);
            // handle error
        } else {
            printf("Can't allocate redis context\n");
        }
        return 0;
    }
    std::cout << "redisConnect global_redis " << &global_redis << std::endl;

    pthread_mutex_init(&global_redis_mutx, NULL);

    /* PUBLISH a key */
    // redisReply *reply = (redisReply *)redisCommand(global_redis, "PUBLISH %s %s", "mychannel", "h3333 world", 3);
    // printf("PUBLISH: %s\n", reply->str);
    // freeReplyObject(reply);

    std::map<std::string, bm::BMNNContextPtr> modelMap;

    std::string bmodel_file = parser.get<std::string>("bmodel");
    std::string output_url = parser.get<std::string>("output");
    std::string config_file = parser.get<std::string>("config");

    int total_num = parser.get<int>("num");
    Config cfg(config_file.c_str());

    // if (!cfg.valid_check(total_num)) {
    //     std::cout << "ERROR:cameras.json config error, please check!" << std::endl;
    //     return -1;
    // }

    const std::vector<CameraConfig> &cameraConfigs = cfg.getCameraConfigs();
    total_num = cameraConfigs.size();
    AppStatis appStatis(total_num);

    int card_num = cfg.cardNums();
    int channel_num_per_card = total_num / card_num;
    int last_channel_num = total_num % card_num == 0 ? 0 : total_num % card_num;

    std::shared_ptr<bm::VideoUIApp> gui;
#if USE_QTGUI
    gui = bm::VideoUIApp::create(argc, argv);
    gui->bootUI(total_num);
#endif

    bm::TimerQueuePtr tqp = bm::TimerQueue::create();
    int start_chan_index = 0;
    std::vector<OneCardInferAppPtr> apps;
    // for (int card_idx = 0; card_idx < card_num; ++card_idx) {
    //     int dev_id = cfg.cardDevId(card_idx);
    //     // load balance
    //     int channel_num = 0;
    //     if (card_idx < last_channel_num) {
    //         channel_num = channel_num_per_card + 1;
    //     } else {
    //         channel_num = channel_num_per_card;
    //     }

    //     bm::BMNNHandlePtr handle = std::make_shared<bm::BMNNHandle>(dev_id);
    //     bm::BMNNContextPtr contextPtr = std::make_shared<bm::BMNNContext>(handle, bmodel_file);
    //     bmlib_log_set_level(BMLIB_LOG_VERBOSE);

    //     int max_batch = parser.get<int>("max_batch");
    //     std::shared_ptr<FaceDetector> det1 = std::make_shared<FaceDetector>(contextPtr, max_batch);
    //     std::shared_ptr<FaceLandmark> det2 = std::make_shared<FaceLandmark>(contextPtr, max_batch);
    //     std::shared_ptr<FaceExtract> det3 = std::make_shared<FaceExtract>(contextPtr, max_batch);
    //     OneCardInferAppPtr appPtr = std::make_shared<OneCardInferApp>(appStatis, gui,
    //                                                                   tqp, contextPtr, output_url,
    //                                                                   start_chan_index, channel_num, 0, 3);
    //     // set detector delegator
    //     appPtr->setDetectorDelegate(0, det1);
    //     appPtr->setDetectorDelegate(1, det2);
    //     appPtr->setDetectorDelegate(2, det3);

    //     start_chan_index += channel_num;

    //     appPtr->start(cfg.cardUrls(card_idx), cfg);
    //     apps.push_back(appPtr);
    // }

    for (int i = 0; i < cameraConfigs.size(); ++i) {
        CameraConfig cConfig = cameraConfigs[i];
        int modelNum = cConfig.models.size();
        int skipFrames = cConfig.skip_frames;

        std::cout << cConfig.models.size() << std::endl;
        std::cout << cConfig.skip_frames << std::endl;
        std::cout << cConfig.url << std::endl;
        std::cout << cConfig.channel_id << std::endl;
        std::cout << cConfig.redis_topic << std::endl;

        OneCardInferAppPtr appPtr = std::make_shared<OneCardInferApp>(appStatis, gui,
                                                                      tqp, output_url,
                                                                      start_chan_index, 1, skipFrames, modelNum);
        for (int j = 0; j < modelNum; ++j) {
            std::cout << cConfig.models[j].name << std::endl;
            std::cout << cConfig.models[j].path << std::endl;
            std::cout << cConfig.models[j].confidence << std::endl;
            std::cout << cConfig.models[j].threshold << std::endl;
            std::cout << cConfig.models[j].nms_threshold << std::endl;
            std::cout << cConfig.models[j].max_batch << std::endl;

            bm::BMNNContextPtr contextPtr = nullptr;
            std::map<std::string, bm::BMNNContextPtr>::iterator iter;
            iter = modelMap.find(cConfig.models[j].path);
            if (iter != modelMap.end()) {
                contextPtr = iter->second;
            } else {
                bm::BMNNHandlePtr handle = std::make_shared<bm::BMNNHandle>(0);
                std::string bmodel_file = cConfig.models[j].path;
                bm::BMNNContextPtr newContextPtr = std::make_shared<bm::BMNNContext>(handle, bmodel_file);
                modelMap.insert(std::pair<std::string, bm::BMNNContextPtr>(cConfig.models[j].path, newContextPtr));
                contextPtr = newContextPtr;
            }

            bmlib_log_set_level(BMLIB_LOG_VERBOSE);

            int max_batch = parser.get<int>("max_batch");
            max_batch = cConfig.models[j].max_batch;

            std::shared_ptr<YoloV5> detector = std::make_shared<YoloV5>(contextPtr, max_batch);

            detector->setConfidence(cConfig.models[j].confidence, cConfig.models[j].threshold, cConfig.models[j].nms_threshold);
            detector->detectorName = cConfig.models[j].name;
            detector->channelId = cConfig.channel_id;
            detector->redisTopic = cConfig.redis_topic;

            std::cout << "here detectorName" << std::endl;
            std::cout << detector->detectorName << std::endl;
            // set detector delegator
            if (j == modelNum - 1) {
                std::cout << "detector->setLastDetector true" << std::endl;
                std::cout << j << std::endl;
                std::cout << modelNum << std::endl;
                detector->setLastDetector(true);
            }
            appPtr->setDetectorDelegate(j, detector);
        }

        // start_chan_index += channel_num;

        appPtr->start(cConfig.url, cfg);
        apps.push_back(appPtr);
    }

    uint64_t timer_id;
    tqp->create_timer(
        1000, [&appStatis]() {
            int ch = 0;
            appStatis.m_stat_imgps->update(appStatis.m_chan_statis[ch]);
            appStatis.m_total_fpsPtr->update(appStatis.m_total_statis);
            double imgps = appStatis.m_stat_imgps->getSpeed();
            double totalfps = appStatis.m_total_fpsPtr->getSpeed();
            std::cout << "[" << bm::timeToString(time(0)) << "] total fps ="
                      << std::setiosflags(std::ios::fixed) << std::setprecision(1) << totalfps
                      << ",ch=" << ch << ": speed=" << imgps << std::endl;
        },
        1, &timer_id);

    tqp->run_loop();

    return 0;
}
