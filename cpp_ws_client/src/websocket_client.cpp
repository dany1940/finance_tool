#include "websocket_client.h"
#include "zmq_kafka_producer.h"
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

WebSocketClient::WebSocketClient(const std::string &exchange, const std::string &url, const std::vector<std::string>& symbols, ZMQKafkaProducer& producer)
    : ssl_ctx(boost::asio::ssl::context::tlsv12_client),
      ws(ioc, ssl_ctx),
      exchange_name(exchange),
      server_url(url),
      stock_symbols(symbols),
      connection_alive(false),
      kafka_producer(producer) {}

void WebSocketClient::connect() {
    boost::asio::ip::tcp::resolver resolver(ioc);
    boost::system::error_code ec;
    std::vector<std::string> common_ports = {"9443", "443", "8080", "2053", "2083", "2087", "2096", "8443"};
    std::string chosen_port;

    std::cout << "🔹 Resolving host for " << exchange_name << ": " << server_url << std::endl;
    std::string clean_host = server_url;

    if (clean_host.find("wss://") == 0) {
        clean_host = clean_host.substr(6);
    } else if (clean_host.find("ws://") == 0) {
        clean_host = clean_host.substr(5);
    }

    std::cout << "🔹 Cleaned host for " << exchange_name << ": " << clean_host << std::endl;

    for (const auto& port : common_ports) {
        std::cout << "🔹 Trying port: " << port << " for " << clean_host << std::endl;
        auto const results = resolver.resolve(clean_host, port, ec);
        if (!ec) {
            chosen_port = port;
            break;
        } else {
            std::cerr << "❌ DNS Resolution Failed for " << exchange_name << " on port " << port << ": " << ec.message() << std::endl;
        }
    }

    if (chosen_port.empty()) {
        std::cerr << "❌ No working port found for " << exchange_name << std::endl;
        return;
    }

    std::cout << "✅ Using port: " << chosen_port << " for " << exchange_name << std::endl;

    auto const results = resolver.resolve(clean_host, chosen_port, ec);
    if (ec) {
        std::cerr << "❌ DNS Resolution Failed for " << exchange_name << ": " << ec.message() << std::endl;
        return;
    }

    boost::asio::connect(ws.next_layer().next_layer(), results.begin(), results.end(), ec);
    if (ec) {
        std::cerr << "❌ TCP Connection Failed for " << exchange_name << ": " << ec.message() << std::endl;
        return;
    }

    if (!SSL_set_tlsext_host_name(ws.next_layer().native_handle(), clean_host.c_str())) {
        boost::system::error_code ec{static_cast<int>(::ERR_get_error()), boost::asio::error::get_ssl_category()};
        std::cerr << "❌ Failed to set SNI Hostname for " << exchange_name << ": " << ec.message() << std::endl;
        return;
    }

    std::cout << "🔒 SNI Hostname Set for " << server_url << std::endl;

    ws.next_layer().handshake(boost::asio::ssl::stream_base::client, ec);
    if (ec) {
        std::cerr << "❌ SSL Handshake Failed for " << exchange_name << ": " << ec.message() << std::endl;
        return;
    }

    std::cout << "🔒 SSL Handshake Successful for " << exchange_name << std::endl;

    std::string path = "/ws";
    ws.handshake(clean_host, path, ec);

    std::cout << "✅ Successfully Connected to " << exchange_name << " WebSocket!" << std::endl;

    connection_alive = true;
    send_subscriptions();
}

void WebSocketClient::send_subscriptions() {
    std::string subscriptionMessage;
    std::string exchangeLower = exchange_name;
    std::transform(exchangeLower.begin(), exchangeLower.end(), exchangeLower.begin(), ::tolower);

    std::cout << "🔹 Sending Subscriptions for " << exchange_name << std::endl;

    if (exchangeLower == "binance") {
        nlohmann::json sub_json;
        sub_json["method"] = "SUBSCRIBE";
        std::vector<std::string> formatted_symbols;

        for (const auto& symbol : stock_symbols) {
            formatted_symbols.push_back(symbol);
        }

        sub_json["params"] = formatted_symbols;
        sub_json["id"] = 1;
        subscriptionMessage = sub_json.dump();
    } else if (exchangeLower == "yahoo finance") {
        subscriptionMessage = nlohmann::json{{"subscribe", stock_symbols}}.dump();
    } else {
        std::cerr << "❌ Unsupported exchange: " << exchange_name << std::endl;
        return;
    }

    send_message(subscriptionMessage);
    std::cout << "📡 Sent Subscription: " << subscriptionMessage << " to " << exchange_name << std::endl;
}


void WebSocketClient::connect_with_retry() {
    std::cerr << "⚠️ Connection lost! Retrying for " << exchange_name << "..." << std::endl;

    int retry_count = 0;
    while (retry_count < 5) {
        try {
            // ✅ Properly close the WebSocket before reconnecting
            boost::system::error_code ec;
            ws.close(boost::beast::websocket::close_code::normal, ec);
            if (ec) {
                std::cerr << "⚠️ WebSocket Close Failed: " << ec.message() << std::endl;
            }

            // ✅ Reset connection state
            connection_alive = false;

            // ✅ Resolve DNS again
            boost::asio::ip::tcp::resolver resolver(ioc);
            auto const results = resolver.resolve(server_url, "443", ec);
            if (ec) {
                throw std::runtime_error("❌ DNS Resolution Failed: " + ec.message());
            }

            // ✅ Reconnect the TCP layer
            boost::asio::connect(ws.next_layer().next_layer(), results.begin(), results.end(), ec);
            if (ec) {
                throw std::runtime_error("❌ TCP Connection Failed: " + ec.message());
            }

            // ✅ Perform SSL handshake again
            ws.next_layer().handshake(boost::asio::ssl::stream_base::client, ec);
            if (ec) {
                throw std::runtime_error("❌ SSL Handshake Failed: " + ec.message());
            }

            // ✅ Perform WebSocket handshake again
            ws.handshake(server_url, "/", ec);
            if (ec) {
                throw std::runtime_error("❌ WebSocket Handshake Failed: " + ec.message());
            }

            std::cout << "✅ Successfully Reconnected to " << exchange_name << " WebSocket!" << std::endl;
            connection_alive = true;

            // ✅ Resend subscriptions
            send_subscriptions();
            return; // Exit retry loop if successful

        } catch (const std::exception &e) {
            retry_count++;
            std::cerr << "❌ Reconnect attempt " << retry_count << " failed for " << exchange_name
                      << ": " << e.what() << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(3));
        }
    }

    std::cerr << "🚨 WebSocket permanently failed for " << exchange_name << std::endl;
}


bool WebSocketClient::is_alive() {
    return connection_alive;
}

std::string WebSocketClient::receive_message() {
    try {
        boost::beast::flat_buffer buffer;
        ws.read(buffer);
        std::string received_data = boost::beast::buffers_to_string(buffer.data());

        try {
            nlohmann::json json_data = nlohmann::json::parse(received_data);

            if (!json_data.is_object() || json_data.empty()) {
                std::cerr << "⚠️ Skipping invalid JSON: " << received_data << std::endl;
                return "";
            }

            for (auto it = json_data.begin(); it != json_data.end();) {
                if (it.value().is_null()) {
                    it = json_data.erase(it);
                } else {
                    ++it;
                }
            }

            std::string validated_data = json_data.dump();
            kafka_producer.send_to_kafka(validated_data);

        } catch (const nlohmann::json::exception &e) {
            std::cerr << "❌ JSON Parse Error: " << e.what() << " | Data: " << received_data << std::endl;
            return "";
        }

        connection_alive = true;
        return received_data;

    } catch (const std::exception &e) {
        std::cerr << "❌ WebSocket Read Error for " << exchange_name << ": " << e.what() << std::endl;
        connection_alive = false;
        connect_with_retry();
        return "";
    }
}

void WebSocketClient::send_heartbeat() {
    while (connection_alive) {
        std::this_thread::sleep_for(std::chrono::seconds(60));
        try {
            ws.ping(boost::beast::websocket::ping_data("keepalive"));
        } catch (const std::exception &e) {
            connection_alive = false;
            connect_with_retry();
        }
    }
}

void WebSocketClient::close() {
    ws.close(boost::beast::websocket::close_code::normal);
    connection_alive = false;
}


void WebSocketClient::send_message(const std::string &message) {
    ws.write(boost::asio::buffer(message));
}




