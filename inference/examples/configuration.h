//
// Created by yuan on 3/12/21.
//

#ifndef INFERENCE_FRAMEWORK_CONFIGURATION_H
#define INFERENCE_FRAMEWORK_CONFIGURATION_H

#include <fstream>
#include <unordered_map>
#include "json/json.h"

struct CardConfig {
    int devid;
    std::vector<std::string> urls;
};

struct CameraModel {
    std::string name;
    std::string path;

    float confidence;
    float threshold;
    float nms_threshold;

    int max_batch;
};

struct CameraConfig {
    std::string url;
    int skip_frames;
    std::string channel_id;
    std::string redis_topic;
    std::vector<CameraModel> models;
};

struct SConcurrencyConfig {
    int thread_num{4};
    int queue_size{4};
    bool blocking{false};

    SConcurrencyConfig() = default;

    SConcurrencyConfig(Json::Value &value) {
        load(value);
    }

    void load(Json::Value &value) {
        thread_num = value["thread_num"].asInt();
        queue_size = value["queue_size"].asInt();
        blocking = value["blocking"].asBool();
    }
};

class Config {
    std::vector<CardConfig> m_cards;
    std::vector<CameraConfig> m_cameras;
    std::unordered_map<std::string, SConcurrencyConfig> m_concurrency;

    void load_config(std::vector<CardConfig> &vctCardConfig, const char *config_file = "cameras.json") {
#if 1
        Json::Reader reader;
        Json::Value json_root;

        std::ifstream in(config_file);
        if (!in.is_open()) {
            printf("Can't open file: %s\n", config_file);
            return;
        }

        if (!reader.parse(in, json_root, false)) {
            return;
        }

        if (json_root["cards"].isNull() || !json_root["cards"].isArray()) {
            in.close();
            return;
        }

        int card_num = json_root["cards"].size();
        for (int card_index = 0; card_index < card_num; ++card_index) {
            Json::Value jsonCard = json_root["cards"][card_index];
            CardConfig card_config;
            card_config.devid = jsonCard["devid"].asInt();
            int camera_num = jsonCard["cameras"].size();
            Json::Value jsonCameras = jsonCard["cameras"];
            for (int i = 0; i < camera_num; ++i) {
                auto json_url_info = jsonCameras[i];
                int chan_num = json_url_info["chan_num"].asInt();
                for (int j = 0; j < chan_num; ++j) {
                    auto url = json_url_info["address"].asString();
                    card_config.urls.push_back(url);
                }
            }

            vctCardConfig.push_back(card_config);
        }

        // load thread_num, queue_size for concurrency
        if (json_root.isMember("pipeline")) {
            Json::Value pipeline_config = json_root["pipeline"];
            maybe_load_concurrency_cfg(pipeline_config, "preprocess");
            maybe_load_concurrency_cfg(pipeline_config, "inference");
            maybe_load_concurrency_cfg(pipeline_config, "postprocess");
        }

        in.close();
#else
        for (int i = 0; i < 2; i++) {
            CardConfig cfg;
            cfg.devid = i;
            std::string url = "/home/yuan/station.avi";
            for (int j = 0; j < 1; j++) {
                cfg.urls.push_back(url);
            }

            m_cards.push_back(cfg);
        }
#endif
    }

    void load_config(std::vector<CardConfig> &vctCardConfig, std::vector<CameraConfig> &vctCameraConfig,
                     const char *config_file = "cameras.json") {
#if 1
        Json::Reader reader;
        Json::Value json_root;

        std::ifstream in(config_file);
        if (!in.is_open()) {
            printf("Can't open file: %s\n", config_file);
            return;
        }

        if (!reader.parse(in, json_root, false)) {
            return;
        }

        if (json_root["cards"].isNull() || !json_root["cards"].isArray()) {
            in.close();
            return;
        }

        int card_num = json_root["cards"].size();
        for (int card_index = 0; card_index < card_num; ++card_index) {
            Json::Value jsonCard = json_root["cards"][card_index];
            int devid = jsonCard["devid"].asInt();
            if (devid != 0) {
                printf("devid error, devid=%d\n", devid);
                return;
            }
            std::string redisTopic = jsonCard["redis_topic"].asString();

            int camera_num = jsonCard["cameras"].size();
            Json::Value jsonCameras = jsonCard["cameras"];
            for (int i = 0; i < camera_num; ++i) {
                auto json_url_info = jsonCameras[i];

                // parse config
                CameraConfig camera_config;
                camera_config.url = json_url_info["address"].asString();
                camera_config.skip_frames = json_url_info["skip_frames"].asInt();
                camera_config.channel_id = json_url_info["channel_id"].asString();
                camera_config.redis_topic = redisTopic;

                Json::Value cameraModels = json_url_info["models"];
                int models_num = json_url_info["models"].size();

                // std::cout << "load config models_num" << std::endl;
                // std::cout << models_num << std::endl;
                for (int j = 0; j < models_num; ++j) {
                    auto model_info = cameraModels[j];
                    CameraModel camera_model;
                    camera_model.confidence = model_info["confidence"].asFloat();
                    camera_model.threshold = model_info["threshold"].asFloat();
                    camera_model.nms_threshold = model_info["nms_threshold"].asFloat();
                    camera_model.name = model_info["name"].asString();
                    camera_model.path = model_info["path"].asString();
                    camera_model.max_batch = model_info["max_batch"].asInt();

                    // std::cout << "load config" << std::endl;
                    // std::cout << camera_model.path << std::endl;
                    // std::cout << camera_model.confidence << std::endl;
                    // std::cout << camera_model.threshold << std::endl;
                    // std::cout << camera_model.name << std::endl;
                    // std::cout << camera_model.nms_threshold << std::endl;
                    camera_config.models.push_back(camera_model);
                }
                vctCameraConfig.push_back(camera_config);
            }
        }

        // load thread_num, queue_size for concurrency
        if (json_root.isMember("pipeline")) {
            Json::Value pipeline_config = json_root["pipeline"];
            maybe_load_concurrency_cfg(pipeline_config, "preprocess");
            maybe_load_concurrency_cfg(pipeline_config, "inference");
            maybe_load_concurrency_cfg(pipeline_config, "postprocess");
        }

        in.close();
#else
        for (int i = 0; i < 2; i++) {
            CardConfig cfg;
            cfg.devid = i;
            std::string url = "/home/yuan/station.avi";
            for (int j = 0; j < 1; j++) {
                cfg.urls.push_back(url);
            }

            m_cards.push_back(cfg);
        }
#endif
    }

public:
    Config(const char *config_file = "cameras.json") {
        load_config(m_cards, m_cameras, config_file);
    }

    int cardNums() {
        return m_cards.size();
    }

    int cardDevId(int index) {
        return m_cards[index].devid;
    }

    const std::vector<std::string> &cardUrls(int index) {
        return m_cards[index].urls;
    }

    bool valid_check(int total) {
        if (m_cards.size() == 0) return false;

        for (int i = 0; i < m_cards.size(); ++i) {
            if (m_cards.size() == 0) return false;
        }

        return true;
    }

    bool maybe_load_concurrency_cfg(Json::Value &json_node, const char *phrase) {
        if (json_node.isMember(phrase)) {
            SConcurrencyConfig cfg(json_node[phrase]);
            m_concurrency.insert(std::make_pair(phrase, cfg));
        }
    }

    bool get_phrase_config(const char *phrase, SConcurrencyConfig &cfg) {
        if (m_concurrency.find(phrase) != m_concurrency.end()) {
            cfg = m_concurrency[phrase];
            return true;
        }
        return false;
    }

    const std::vector<CameraConfig> &getCameraConfigs() {
        return m_cameras;
    }
};

struct AppStatis {
    int m_channel_num;
    bm::StatToolPtr m_stat_imgps;
    bm::StatToolPtr m_total_fpsPtr;
    uint64_t *m_chan_statis;
    uint64_t m_total_statis = 0;
    std::mutex m_statis_lock;

    AppStatis(int num) :
        m_channel_num(num) {
        m_stat_imgps = bm::StatTool::create(5);
        m_total_fpsPtr = bm::StatTool::create(5);
        m_chan_statis = new uint64_t[m_channel_num];
        assert(m_chan_statis != nullptr);
    }

    ~AppStatis() {
        delete[] m_chan_statis;
    }
};

#endif // INFERENCE_FRAMEWORK_CONFIGURATION_H
