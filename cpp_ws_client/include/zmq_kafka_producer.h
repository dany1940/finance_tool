#ifndef ZMQ_KAFKA_PRODUCER_H
#define ZMQ_KAFKA_PRODUCER_H

#include "zmq_handler.h"
#include <rdkafka.h>
#include <string>
#include <unordered_map>
#include <chrono>

class ZMQKafkaProducer {
public:
    ZMQKafkaProducer(const std::string &topic);
    ~ZMQKafkaProducer();

    void send_to_kafka(const std::string &message);

private:
    rd_kafka_t *producer;
    rd_kafka_topic_t *kafka_topic;
    rd_kafka_conf_t *conf;
    ZMQHandler *zmq_handler;

    std::unordered_map<std::string, std::chrono::steady_clock::time_point> message_timestamps;
};

#endif // ZMQ_KAFKA_PRODUCER_H
