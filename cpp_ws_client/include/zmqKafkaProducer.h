#ifndef ZMQ_KAFKA_PRODUCER_H
#define ZMQ_KAFKA_PRODUCER_H

#include "zmqHandler.h"
#include <rdkafka.h>
#include <string>
#include <unordered_map>
#include <chrono>
#include <nlohmann/json.hpp>


using namespace std;
using namespace chrono;
using namespace nlohmann;

// ZMQKafkaProducer class definition
class ZMQKafkaProducer {
public:
    ZMQKafkaProducer();
    ~ZMQKafkaProducer();

    void sendToKafka(const string &exchange_name, const string &message);
    string generateUniqueId(const string &exchangeName, const json &rawData);
    int64_t getTimestamp(const json &rawData);

private:
    rd_kafka_t *producer;
    rd_kafka_conf_t *conf;
    ZMQHandler *zmqHandler;
    unordered_map<string, rd_kafka_topic_t*> kafkaTopics;

    unordered_map<string, steady_clock::time_point> message_timestamps;
};

#endif // ZMQ_KAFKA_PRODUCER_H
