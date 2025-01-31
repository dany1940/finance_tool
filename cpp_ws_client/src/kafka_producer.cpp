#include "kafka_producer.h"
#include <librdkafka/rdkafka.h>
#include <iostream>

using namespace std;

void produce_to_kafka(const string &topic, const string &message) {
    rd_kafka_t *producer;
    rd_kafka_conf_t *conf = rd_kafka_conf_new();
    rd_kafka_conf_set(conf, "bootstrap.servers", "localhost:9092", NULL, 0);
    producer = rd_kafka_new(RD_KAFKA_PRODUCER, conf, NULL, 0);

    rd_kafka_producev(producer, RD_KAFKA_V_TOPIC(topic.c_str()),
                      RD_KAFKA_V_VALUE(message.c_str(), message.size()),
                      RD_KAFKA_V_END);

    rd_kafka_poll(producer, 0);
    rd_kafka_flush(producer, 5000);
    rd_kafka_destroy(producer);
}
