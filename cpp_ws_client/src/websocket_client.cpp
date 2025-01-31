#include "websocket_client.h"
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/system/error_code.hpp>

WebSocketClient::WebSocketClient(const std::string &url)
    : ws(ioc), server_url(url), connection_alive(false) {}

void WebSocketClient::connect() {
    boost::asio::ip::tcp::resolver resolver(ioc);
    boost::system::error_code ec;
    auto const results = resolver.resolve(server_url, "443", ec);

    if (ec) {
        std::cerr << "❌ WebSocket DNS Resolution Failed: " << ec.message() << std::endl;
        throw std::runtime_error("DNS resolution failed");
    }

    boost::asio::connect(ws.next_layer(), results.begin(), results.end());
    ws.handshake(server_url, "/");
    std::cout << "✅ Connected to WebSocket: " << server_url << std::endl;

    connection_alive = true;  // ✅ Set connection as alive
}


void WebSocketClient::connect_with_retry() {
    int retries = 5;
    while (retries > 0) {
        try {
            connect();
            return;
        } catch (const std::exception &e) {
            std::cerr << "❌ Connection failed: " << e.what() << ". Retrying in 3 seconds..." << std::endl;
            retries--;
            std::this_thread::sleep_for(std::chrono::seconds(3));
        }
    }
    std::cerr << "❌ WebSocket connection permanently failed!" << std::endl;
}

void WebSocketClient::send_message(const std::string &message) {
    ws.write(boost::asio::buffer(message));
}

void WebSocketClient::receive_message() {
    boost::beast::flat_buffer buffer;
    ws.read(buffer);
    std::cout << "📩 Received: " << boost::beast::buffers_to_string(buffer.data()) << std::endl;
    connection_alive = true;  // ✅ Mark connection as alive if a message is received
}

void WebSocketClient::send_heartbeat() {
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(5));  // ✅ Send heartbeat every 5 seconds

        if (connection_alive) {
            send_message("ping");  // ✅ Send a heartbeat message
            std::cout << "💓 Sent heartbeat to server" << std::endl;
        } else {
            std::cerr << "⚠️ Connection lost! Retrying..." << std::endl;
            connect_with_retry();
        }
    }
}

bool WebSocketClient::is_alive() {
    return connection_alive;
}

void WebSocketClient::close() {
    ws.close(boost::beast::websocket::close_code::normal);
    connection_alive = false;
}
