#include <iostream>
#include <thread>
#include "websocket_client.h"
#include "zmq_kafka_producer.h"

int main() {
    ZMQKafkaProducer kafka_producer("stock_data");  // ✅ Instantiate producer

    // ✅ Use explicit WebSocket URLs
    std::vector<std::tuple<std::string, std::string, std::vector<std::string>>> exchanges = {
        {"Yahoo Finance", "wss://streamer.finance.yahoo.com", {"AAPL"}},
        {"Binance", "wss://stream.binance.com", {"btcusdt@trade"}}
    };

    std::vector<std::thread> threads;
    std::vector<std::shared_ptr<WebSocketClient>> clients;

    for (const auto& [exchange_name, exchange_url, stocks] : exchanges) {
        std::cout << "🔹 Connecting to exchange: " << exchange_name << " (" << exchange_url << ")" << std::endl;

        auto client = std::make_shared<WebSocketClient>(exchange_name, exchange_url, stocks, kafka_producer);
        clients.push_back(client);

        threads.emplace_back([client]() {
            client->connect();
            while (client->is_alive()) {
                client->receive_message();
            }
        });
    }

    for (auto& client : clients) {
        threads.emplace_back(&WebSocketClient::send_heartbeat, client);
    }

    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    return 0;
}
