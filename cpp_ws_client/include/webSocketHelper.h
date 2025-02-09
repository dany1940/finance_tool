#ifndef WEBSOCKETHELPER_H
#define WEBSOCKETHELPER_H
#include <webSocketClient.h>
#include <vector>
#include <memory>
#include <thread>
#include "zmqKafkaProducer.h"

// Function to start WebSocket clients for each exchange
void startWebSocketClients(ZMQKafkaProducer &kafkaProducer, std::vector<std::thread> &threads, std::vector<std::shared_ptr<WebSocketClient>> &clients);

#endif  // WEBSOCKETHELPER_H
