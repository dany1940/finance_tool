#ifndef KAFKA_PRODUCER_H
#define KAFKA_PRODUCER_H

#include <string>

using namespace std;

void produce_to_kafka(const string &topic, const string &message);

#endif // KAFKA_PRODUCER_H
