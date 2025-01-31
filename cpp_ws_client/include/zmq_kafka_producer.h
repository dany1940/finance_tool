#ifndef ZMQ_KAFKA_PRODUCER_H
#define ZMQ_KAFKA_PRODUCER_H

#include "zmq_handler.h"
#include <rdkafka.h>
#include <string>

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
};

#endif // ZMQ_KAFKA_PRODUCER_H
