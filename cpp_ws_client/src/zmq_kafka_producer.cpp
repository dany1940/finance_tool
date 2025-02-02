
#include "zmq_kafka_producer.h"
#include <iostream>
#include <sstream>
#include <chrono>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ✅ Constructor: Initializes Kafka & ZeroMQ
ZMQKafkaProducer::ZMQKafkaProducer(const std::string &topic) {
    conf = rd_kafka_conf_new();

    // ✅ Ensure broker connection
    rd_kafka_conf_set(conf, "bootstrap.servers", "localhost:9092", NULL, 0);
    rd_kafka_conf_set(conf, "queue.buffering.max.ms", "10", NULL, 0);  // ✅ Reduce latency

    producer = rd_kafka_new(RD_KAFKA_PRODUCER, conf, NULL, 0);
    kafka_topic = rd_kafka_topic_new(producer, topic.c_str(), NULL);
    zmq_handler = new ZMQHandler("127.0.0.1", 5555);

    if (!producer || !kafka_topic) {
        std::cerr << "❌ Kafka Producer Initialization Failed!" << std::endl;
        exit(1);
    }
    std::cout << "✅ Kafka Producer Initialized!" << std::endl;
}

// ✅ Destructor: Cleans up resources
ZMQKafkaProducer::~ZMQKafkaProducer() {
    rd_kafka_flush(producer, 1000);
    rd_kafka_topic_destroy(kafka_topic);
    rd_kafka_destroy(producer);
    delete zmq_handler;
}

// ✅ Ensure Deduplication & Timely Sending
void ZMQKafkaProducer::send_to_kafka(const std::string &message) {
    try {
        json data = json::parse(message);

        // ✅ Ensure required fields exist
        if (!data.contains("E") || !data.contains("t") || !data.contains("s")) {
            std::cerr << "⚠️ Skipping malformed trade data: " << message << std::endl;
            return;
        }

        // ✅ Generate a unique ID using timestamp (E) + trade ID (t) + symbol (s)
        std::string message_id = std::to_string(data["E"].get<uint64_t>()) + "_" +
                                 std::to_string(data["t"].get<uint64_t>()) + "_" +
                                 data["s"].get<std::string>();

        auto now = std::chrono::steady_clock::now();
        if (message_timestamps.count(message_id)) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - message_timestamps[message_id]
            ).count();

            if (elapsed < 500) {  // ✅ Prevent duplicates within 500ms
                std::cerr << "⚠️ Skipping duplicate message: " << message_id << std::endl;
                return;
            }
        }

        // ✅ Store timestamp for deduplication
        message_timestamps[message_id] = now;

        // ✅ Send message to Kafka
        rd_kafka_produce(kafka_topic, RD_KAFKA_PARTITION_UA, RD_KAFKA_MSG_F_COPY,
                         (void *)message.c_str(), message.size(),
                         NULL, 0, NULL);
        std::cout << "📩 Sent to Kafka: " << message << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "❌ Kafka Send Error: " << e.what() << std::endl;
    }
}
