#include "webSocketHelper.h"
#include "spdlog/spdlog.h"
#include "webSocketClient.h"
#include "common.h"

void startWebSocketClients(ZMQKafkaProducer &kafkaProducer, std::vector<std::thread> &threads, std::vector<std::shared_ptr<WebSocketClient>> &clients) {
    for (const auto& [exchangeName, data] : dataSources) {
        const std::string& exchangeUrl = data.first;
        const std::vector<std::string>& stocks = data.second;

        spdlog::info("Connecting to exchange: {} ({})", exchangeName, exchangeUrl);

        // Create WebSocket client for the exchange
        auto client = std::make_shared<WebSocketClient>(exchangeName, exchangeUrl, stocks, kafkaProducer);
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
}
