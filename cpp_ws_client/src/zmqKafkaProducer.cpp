#include "zmqKafkaProducer.h"
#include <iostream>
#include <sstream>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <chrono>

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
        error("Kafka Producer Initialization Failed!");
        exit(1);
    }
    info("Kafka Producer Initialized!");
}

// Cleans up resources
ZMQKafkaProducer::~ZMQKafkaProducer() {
    rd_kafka_flush(producer, 1000);
    rd_kafka_destroy(producer);
    delete zmqHandler;
    info("Kafka Producer Shutdown Complete!");
}

// Generates a unique ID dynamically
string ZMQKafkaProducer::generateUniqueId(const string &exchangeName, const json &rawData) {
    if (rawData.contains("trade_id")) {
        return exchangeName + "_" + to_string(rawData["trade_id"].get<int64_t>());
    } else if (rawData.contains("t")) {
        return exchangeName + "_" + to_string(rawData["t"].get<int64_t>());
    }
    return exchangeName + "_" + to_string(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());
}

// Gets the best available timestamp

int64_t ZMQKafkaProducer::getTimestamp(const json &rawData) {
    if (rawData.contains("T")) { // Binance timestamp (already in milliseconds)
        return rawData["T"].get<int64_t>();
    } else if (rawData.contains("time")) { // Coinbase timestamp (ISO 8601)
        string isoTime = rawData["time"].get<string>();

        std::tm tm = {};
        std::istringstream ss(isoTime);
        ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");

        // Convert to time_t (seconds since epoch)
        time_t timeSinceEpoch = mktime(&tm);

        // Convert to milliseconds
        int64_t milliseconds = static_cast<int64_t>(timeSinceEpoch) * 1000;

        return milliseconds;
    } else {
        return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count(); // Default: Current time
    }
}

// Sends message to Kafka with a unique ID and timestamp
void ZMQKafkaProducer::sendToKafka(const string &exchangeName, const string &message) {
    string exchangeLower = exchangeName;
    transform(exchangeLower.begin(), exchangeLower.end(), exchangeLower.begin(), ::tolower);

    try {
        json rawData = json::parse(message);
        rawData["exchange"] = exchangeName;

        // Assign unique ID and timestamps
        rawData["uniqueId"] = generateUniqueId(exchangeName, rawData);
        rawData["timestamp"] = getTimestamp(rawData);
        rawData["timestampSentToKafka"] = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

        string modifiedMessage = rawData.dump();
        info("Modified message with unique ID: {}", modifiedMessage);

        // Define Kafka topics per exchange
        static const unordered_map<string, string> exchangeToTopic = {
            {"coinbase", "coinbase_ticker"},
            {"binance", "binance_ticker"}
        };

        auto it = exchangeToTopic.find(exchangeLower);
        if (it == exchangeToTopic.end()) {
            warn("Unsupported exchange: {}", exchangeLower);
            return;
        }

        string kafkaTopicName = it->second;

        if (kafkaTopics.find(kafkaTopicName) == kafkaTopics.end()) {
            kafkaTopics[kafkaTopicName] = rd_kafka_topic_new(producer, kafkaTopicName.c_str(), NULL);
            if (!kafkaTopics[kafkaTopicName]) {
                error("Kafka Topic Creation Failed: {}", kafkaTopicName);
                return;
            }
            info("Kafka Topic Created: {}", kafkaTopicName);
        }

        int err = rd_kafka_produce(kafkaTopics[kafkaTopicName], RD_KAFKA_PARTITION_UA, RD_KAFKA_MSG_F_COPY,
                                   (void *)modifiedMessage.c_str(), modifiedMessage.size(),
                                   NULL, 0, NULL);

        if (err == -1) {
            error("Kafka Send Error: {}", rd_kafka_err2str(rd_kafka_last_error()));
        } else {
            info("Sent to Kafka ({}): {}", kafkaTopicName, modifiedMessage);
        }

        rd_kafka_poll(producer, 0);
        info("Kafka Poll Completed for topic: {}", kafkaTopicName);

    } catch (const exception &e) {
        error("Kafka Send Error: {}", e.what());
    }
}
