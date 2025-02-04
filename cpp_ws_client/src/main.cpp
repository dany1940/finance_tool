#include <iostream>
#include <thread>
#include <vector>
#include <memory>
#include "webSocketClient.h"
#include "zmqKafkaProducer.h"
#include <spdlog/spdlog.h> // Use spdlog for logging

using namespace std;
using namespace spdlog;

int main() {
    // Initialize Kafka Producer
    ZMQKafkaProducer kafkaProducer;

    // Define WebSocket connections for different exchanges
    vector<tuple<string, string, vector<string>>> exchanges = {
        {"Yahoo Finance", "wss://streamer.finance.yahoo.com", {"AAPL"}},
        {"Binance", "wss://stream.binance.com", {"btcusdt@trade"}},
        {"Coinbase", "wss://ws-feed.exchange.coinbase.com", {"BTC-USD"}}
    };

    vector<thread> threads;
    vector<shared_ptr<WebSocketClient>> clients;

    // Create WebSocket clients for each exchange
    for (const auto& [exchangeName, exchangeUrl, stocks] : exchanges) {
        info("Connecting to exchange: {} ({})", exchangeName, exchangeUrl);

        auto client = make_shared<WebSocketClient>(exchangeName, exchangeUrl, stocks, kafkaProducer);
        clients.push_back(client);

        // Launch WebSocket connection in a separate thread
        threads.emplace_back([client]() {
            client->connect();
            while (client->isAlive()) {
                client->receiveMessage();
            }
        });
    }

    // Start heartbeat threads for connection monitoring
    for (auto& client : clients) {
        threads.emplace_back(&WebSocketClient::sendHeartbeat, client);
    }

    // Wait for all threads to finish
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    return 0;
}
