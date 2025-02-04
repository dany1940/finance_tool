#include "webSocketClient.h"
#include "zmqKafkaProducer.h"
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/ssl.hpp>
#include <iostream>
#include <sstream>
#include <thread>
#include <regex>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

// Namespace Aliases
namespace asio = boost::asio;
namespace beast = boost::beast;
namespace syst= boost::system;
namespace websocket = beast::websocket;
namespace ip = asio::ip;
namespace log = spdlog;
namespace stdx = std;

WebSocketClient::WebSocketClient(const stdx::string &exchange, const stdx::string &url, const stdx::vector<stdx::string>& symbols, ZMQKafkaProducer& producer)
    : sslCtx(asio::ssl::context::tlsv12_client),
      webSocket(ioc, sslCtx),
      exchangeName(exchange),
      serverUrl(url),
      stockSymbols(symbols),
      connectionAlive(false),
      kafkaProducer(producer) {}

void WebSocketClient::connect() {
    ip::tcp::resolver resolver(ioc);
    syst::error_code ec;
    stdx::map<stdx::string, stdx::string> portMapping = {
        {"Yahoo Finance", "443"},
        {"Binance", "9443"},
        {"Coinbase", "443"},
    };

    stdx::string chosenPort = portMapping[exchangeName];

    log::info("Resolving host for {}: {}", exchangeName, serverUrl);
    stdx::string cleanHost = serverUrl;

    if (cleanHost.find("wss://") == 0) {
        cleanHost = cleanHost.substr(6);
    } else if (cleanHost.find("ws://") == 0) {
        cleanHost = cleanHost.substr(5);
    }

    log::info("Cleaned host for {}: {}", exchangeName, cleanHost);
    log::info("Using port: {} for {}", chosenPort, exchangeName);

    auto const results = resolver.resolve(cleanHost, chosenPort, ec);
    if (ec) {
        log::error("DNS Resolution Failed for {}: {}", exchangeName, ec.message());
        return;
    }

    asio::connect(webSocket.next_layer().next_layer(), results.begin(), results.end(), ec);
    if (ec) {
        log::error("TCP Connection Failed for {}: {}", exchangeName, ec.message());
        return;
    }

    if (!SSL_set_tlsext_host_name(webSocket.next_layer().native_handle(), cleanHost.c_str())) {
        log::error("Failed to set SNI Hostname for {}", exchangeName);
        return;
    }

    log::info("SNI Hostname Set for {}", serverUrl);

    webSocket.next_layer().handshake(asio::ssl::stream_base::client, ec);
    if (ec) {
        log::error("SSL Handshake Failed for {}: {}", exchangeName, ec.message());
        return;
    }

    log::info("SSL Handshake Successful for {}", exchangeName);

    stdx::string path = "/ws";
    webSocket.handshake(cleanHost, path, ec);

    log::info("Successfully Connected to {} WebSocket!", exchangeName);

    connectionAlive = true;
    sendSubscriptions();
}

void WebSocketClient::sendSubscriptions() {
    static const stdx::unordered_map<stdx::string, stdx::function<stdx::string(const stdx::vector<stdx::string>&)>> subscriptionFormats = {
        {"binance", [](const stdx::vector<stdx::string>& symbols) {
            nlohmann::json subJson;
            subJson["method"] = "SUBSCRIBE";
            subJson["params"] = symbols;
            subJson["id"] = 1;
            return subJson.dump();
        }},
        {"yahoo finance", [](const stdx::vector<stdx::string>& symbols) {
            return nlohmann::json{{"subscribe", symbols}}.dump();
        }},
        {"coinbase", [](const stdx::vector<stdx::string>& symbols) {
            nlohmann::json subJson;
            subJson["type"] = "subscribe";
            nlohmann::json channels = nlohmann::json::array();
            channels.push_back({{"name", "ticker"}, {"product_ids", symbols}});
            subJson["channels"] = channels;
            return subJson.dump();
        }}
    };

    stdx::string exchangeLower = exchangeName;
    stdx::transform(exchangeLower.begin(), exchangeLower.end(), exchangeLower.begin(), ::tolower);

    auto it = subscriptionFormats.find(exchangeLower);
    if (it != subscriptionFormats.end()) {
        stdx::string subscriptionMessage = it->second(stockSymbols);
        sendMessage(subscriptionMessage);
        log::info("Sent Subscription: {} to {}", subscriptionMessage, exchangeName);
    } else {
        log::warn("Unsupported exchange: {}", exchangeName);
    }
}

void WebSocketClient::connectWithRetry() {
    log::warn("Connection lost! Retrying for {}...", exchangeName);

    int retryCount = 0;
    while (retryCount < 5) {
        try {
            syst::error_code ec;
            webSocket.close(websocket::close_code::normal, ec);
            if (ec) {
                log::warn("WebSocket Close Failed: {}", ec.message());
            }

            connectionAlive = false;

            ip::tcp::resolver resolver(ioc);
            auto const results = resolver.resolve(serverUrl, "443", ec);
            if (ec) {
                throw stdx::runtime_error("DNS Resolution Failed: " + ec.message());
            }

            asio::connect(webSocket.next_layer().next_layer(), results.begin(), results.end(), ec);
            if (ec) {
                throw stdx::runtime_error("TCP Connection Failed: " + ec.message());
            }

            webSocket.next_layer().handshake(asio::ssl::stream_base::client, ec);
            if (ec) {
                throw stdx::runtime_error("SSL Handshake Failed: " + ec.message());
            }

            webSocket.handshake(serverUrl, "/", ec);
            if (ec) {
                throw stdx::runtime_error("WebSocket Handshake Failed: " + ec.message());
            }

            log::info("Successfully Reconnected to {} WebSocket!", exchangeName);
            connectionAlive = true;

            sendSubscriptions();
            return;

        } catch (const stdx::exception &e) {
            retryCount++;
            log::error("Reconnect attempt {} failed for {}: {}", retryCount, exchangeName, e.what());
            stdx::this_thread::sleep_for(stdx::chrono::seconds(3));
        }
    }

    log::error("WebSocket permanently failed for {}", exchangeName);
}

bool WebSocketClient::isAlive() {
    return connectionAlive;
}

stdx::string WebSocketClient::receiveMessage() {
    try {
        beast::flat_buffer buffer;
        webSocket.read(buffer);
        stdx::string receivedData = beast::buffers_to_string(buffer.data());

        try {
            nlohmann::json jsonData = nlohmann::json::parse(receivedData);

            if (!jsonData.is_object() || jsonData.empty()) {
                log::warn("Skipping invalid JSON: {}", receivedData);
                return "";
            }

            for (auto it = jsonData.begin(); it != jsonData.end();) {
                if (it.value().is_null()) {
                    it = jsonData.erase(it);
                } else {
                    ++it;
                }
            }

            stdx::string validatedData = jsonData.dump();
            kafkaProducer.sendToKafka(exchangeName, validatedData);

        } catch (const nlohmann::json::exception &e) {
            log::error("JSON Parse Error: {} | Data: {}", e.what(), receivedData);
            return "";
        }

        connectionAlive = true;
        return receivedData;

    } catch (const stdx::exception &e) {
        log::error("WebSocket Read Error for {}: {}", exchangeName, e.what());
        connectionAlive = false;
        connectWithRetry();
        return "";
    }
}
// Sends periodic heartbeat messages to keep the connection alive
void WebSocketClient::sendHeartbeat() {
    while (connectionAlive) {
        this_thread::sleep_for(chrono::seconds(60));
        try {
            webSocket.ping(websocket::ping_data("keepalive"));
        } catch (const exception &e) {
            connectionAlive = false;
            connectWithRetry();
        }
    }
}

void WebSocketClient::sendMessage(const stdx::string &message) {
    webSocket.write(asio::buffer(message));
}
