#include "zmq_kafka_producer.h"
#include <iostream>

ZMQKafkaProducer::ZMQKafkaProducer(const std::string &topic) {
    // Initialize Kafka
    conf = rd_kafka_conf_new();
    rd_kafka_conf_set(conf, "bootstrap.servers", "localhost:9092", NULL, 0);

    producer = rd_kafka_new(RD_KAFKA_PRODUCER, conf, NULL, 0);
    if (!producer) {
        throw std::runtime_error("❌ Failed to create Kafka producer.");
    }

    kafka_topic = rd_kafka_topic_new(producer, topic.c_str(), NULL);

    // Initialize ZeroMQ
    zmq_handler = new ZMQHandler("127.0.0.1", 5555);
}

void ZMQKafkaProducer::send_to_kafka(const std::string &message) {
    // Send via ZeroMQ first
    zmq_handler->send_message(message);

    // Send to Kafka for durability
    rd_kafka_produce(kafka_topic, RD_KAFKA_PARTITION_UA,
                     RD_KAFKA_MSG_F_COPY, (void *)message.c_str(),
                     message.size(), NULL, 0, NULL);

    rd_kafka_poll(producer, 0);
    std::cout << "✅ Kafka Message Sent: " << message << std::endl;
}

ZMQKafkaProducer::~ZMQKafkaProducer() {
    rd_kafka_topic_destroy(kafka_topic);
    rd_kafka_destroy(producer);
    delete zmq_handler;
}
