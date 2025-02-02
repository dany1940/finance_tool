#ifndef WEBSOCKET_CLIENT_H
#define WEBSOCKET_CLIENT_H

#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/asio/ssl/stream.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/connect.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include "zmq_kafka_producer.h"

namespace ssl = boost::asio::ssl;
namespace websocket = boost::beast::websocket;
namespace ip = boost::asio::ip;

class WebSocketClient {
public:
    explicit WebSocketClient(const std::string &exchange,
                             const std::string &url,
                             const std::vector<std::string>& symbols,
                             ZMQKafkaProducer &producer);  // ✅ Ensure Kafka Producer

    void connect();
    void connect_with_retry();
    void send_message(const std::string &message);
    std::string receive_message();
    void send_heartbeat();
    bool is_alive();
    void close();
    void send_subscriptions();
    std::string get_stock_list();

private:
    boost::asio::io_context ioc;
    ssl::context ssl_ctx;
    websocket::stream<ssl::stream<ip::tcp::socket>> ws;
    std::string exchange_name;
    std::string server_url;
    std::vector<std::string> stock_symbols;
    bool connection_alive;
    ZMQKafkaProducer &kafka_producer;  // ✅ Ensure Kafka Producer Reference
};

#endif  // WEBSOCKET_CLIENT_H
