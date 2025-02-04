#include "zmqKafkaProducer.h"
#include <iostream>
#include <sstream>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

using namespace std;
using namespace chrono;
using namespace nlohmann;
using namespace spdlog;

// Initializes Kafka & ZeroMQ
ZMQKafkaProducer::ZMQKafkaProducer() {
    conf = rd_kafka_conf_new();
    rd_kafka_conf_set(conf, "bootstrap.servers", "localhost:9092", NULL, 0);
    rd_kafka_conf_set(conf, "queue.buffering.max.ms", "10", NULL, 0);

    producer = rd_kafka_new(RD_KAFKA_PRODUCER, conf, NULL, 0);
    zmqHandler = new ZMQHandler("127.0.0.1", 5555);

    if (!producer) {
        error("❌ Kafka Producer Initialization Failed!");
        exit(1);
    }
    info("✅ Kafka Producer Initialized!");
}

// Cleans up resources
ZMQKafkaProducer::~ZMQKafkaProducer() {
    rd_kafka_flush(producer, 1000);
    rd_kafka_destroy(producer);
    delete zmqHandler;
    info("✅ Kafka Producer Shutdown Complete!");
}

// Sends message to Kafka under the appropriate topic
void ZMQKafkaProducer::sendToKafka(const string &exchangeName, const string &message) {
    string exchangeLower = exchangeName;
    transform(exchangeLower.begin(), exchangeLower.end(), exchangeLower.begin(), ::tolower);

    try {
        json rawData = json::parse(message);

        // Define Kafka topics per exchange
        static const unordered_map<string, string> exchangeToTopic = {
            {"coinbase", "coinbase_ticker"},
            {"binance", "binance_ticker"},
            {"yahoo finance", "yahoo_ticker"}
        };

        // Check if exchange is supported
        auto it = exchangeToTopic.find(exchangeLower);
        if (it == exchangeToTopic.end()) {
            warn("⚠️ Unsupported exchange: {}", exchangeLower);
            return;
        }

        string kafkaTopicName = it->second;

        // Check if topic is already created
        if (kafkaTopics.find(kafkaTopicName) == kafkaTopics.end()) {
            kafkaTopics[kafkaTopicName] = rd_kafka_topic_new(producer, kafkaTopicName.c_str(), NULL);
            if (!kafkaTopics[kafkaTopicName]) {
                error("❌ Kafka Topic Creation Failed: {}", kafkaTopicName);
                return;
            }
            info("✅ Kafka Topic Created: {}", kafkaTopicName);
        }

        // Send message to Kafka
        int err = rd_kafka_produce(kafkaTopics[kafkaTopicName], RD_KAFKA_PARTITION_UA, RD_KAFKA_MSG_F_COPY,
                                   (void *)message.c_str(), message.size(),
                                   NULL, 0, NULL);

        if (err == -1) {
            error("❌ Kafka Send Error: {}", rd_kafka_err2str(rd_kafka_last_error()));
        } else {
            info("📩 Sent to Kafka ({}): {}", kafkaTopicName, message);
        }

        // Poll Kafka to ensure message is delivered
        rd_kafka_poll(producer, 0);
        info("✅ Kafka Poll Completed for topic: {}", kafkaTopicName);

    } catch (const exception &e) {
        error("❌ Kafka Send Error: {}", e.what());
    }
}
