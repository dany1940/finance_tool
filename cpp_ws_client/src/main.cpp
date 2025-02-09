#include <iostream>
#include <thread>
#include <vector>
#include <memory>
#include "common.h" // Include the shared dataSources map
#include "webSocketClient.h" // WebSocket client handling (already refactored)
#include "webSocketHelper.h"
#include "zmqKafkaProducer.h" // Kafka Producer
#include <spdlog/spdlog.h> // Logging

using namespace std;
using namespace spdlog;

int main() {
    // Initialize Kafka Producer
    ZMQKafkaProducer kafkaProducer;

    // Create threads and clients for WebSocket connections
    vector<thread> threads;
    vector<shared_ptr<WebSocketClient>> clients;

    // Use the helper function to start WebSocket clients
    startWebSocketClients(kafkaProducer, threads, clients);

    // Wait for all threads to finish
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    return 0;
}

