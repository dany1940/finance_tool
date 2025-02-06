#ifndef WEBSOCKET_CLIENT_H
#define WEBSOCKET_CLIENT_H

#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/asio/ssl/stream.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/io_context.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <spdlog/spdlog.h>
#include "zmqKafkaProducer.h"

using namespace std;  // Keep standard namespace
namespace ssl = boost::asio::ssl;
namespace websocket = boost::beast::websocket;
namespace ip = boost::asio::ip;
namespace IoContext = boost::asio;  // Alias for io_context
namespace log = spdlog;  // Alias for logging

class WebSocketClient {
public:
    WebSocketClient(const string &exchange, const string &url, const vector<string>& symbols, ZMQKafkaProducer &producer);

    void connect();             // Establish WebSocket connection
    void connectWithRetry();    // Retry connection if lost
    void sendMessage(const string &message);  // Send message via WebSocket
    string receiveMessage();    // Receive WebSocket messages
    void sendHeartbeat();       // Maintain connection with heartbeat
    bool isAlive();             // Check connection status
    void close();               // Close WebSocket connection
    void sendSubscriptions();   // Subscribe to stock symbols
    string getStockList();      // Get subscribed stocks
    void authenticateAndSubscribe(); //authentificate exchanges


private:
    IoContext::io_context ioc;  // Boost ASIO IO context
    ssl::context sslCtx;
    websocket::stream<ssl::stream<ip::tcp::socket>> webSocket;
    string exchangeName;
    string serverUrl;
    vector<string> stockSymbols;
    bool connectionAlive;
    ZMQKafkaProducer &kafkaProducer;  // Reference to Kafka producer
};

#endif  // WEBSOCKET_CLIENT_H
