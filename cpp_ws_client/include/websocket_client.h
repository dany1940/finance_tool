
#ifndef WEBSOCKET_CLIENT_H
#define WEBSOCKET_CLIENT_H

#include <boost/beast.hpp>
#include <boost/asio.hpp>
#include <string>
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>

class WebSocketClient {
public:
    WebSocketClient(const std::string &url);
    void connect();
    void connect_with_retry();
    void send_message(const std::string &message);
    void receive_message();
    void send_heartbeat();  // ✅ New heartbeat function
    void close();
    bool is_alive();  // ✅ Check if the connection is alive

private:
    boost::asio::io_context ioc;
    boost::beast::websocket::stream<boost::asio::ip::tcp::socket> ws;
    std::string server_url;
    std::atomic<bool> connection_alive;  // ✅ Track connection status
};

#endif // WEBSOCKET_CLIENT_H

