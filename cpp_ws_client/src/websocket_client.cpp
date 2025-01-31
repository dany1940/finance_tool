#include "websocket_client.h"
#include <boost/asio/connect.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <iostream>

using namespace std;
using namespace boost::asio;
using namespace boost::beast;

// **WebSocket Connection Function**
void connect_to_exchange(const string &exchange_name, const string &exchange_url) {
    io_context ioc;
    tcp::resolver resolver{ioc};
    websocket::stream<tcp::socket> ws{ioc};

    try {
        // Resolve hostname and connect
        auto const results = resolver.resolve(exchange_url, "443");
        connect(ws.next_layer(), results.begin(), results.end());

        // Perform WebSocket handshake
        ws.handshake(exchange_url, "/");
        cout << "✅ Connected to " << exchange_name << endl;

        // Subscribe to stock data
        ws.write(buffer("{\"subscribe\": [\"AAPL\"]}"));

        // Read messages
        for (;;) {
            flat_buffer buffer;
            ws.read(buffer);
            cout << "📩 Received: " << buffers_to_string(buffer.data()) << endl;
        }
    } catch (exception &e) {
        cerr << "❌ WebSocket Error: " << e.what() << endl;
    }
}

// **Heartbeat Function**
void send_heartbeat(const string &exchange) {
    cout << "💓 Sending heartbeat to " << exchange << endl;
}

// **Send Data to FastAPI**
void send_to_fastapi(const string &exchange, const string &payload) {
    (void)exchange;
    cout << "📤 Sending data to FastAPI: " << payload << endl;
}
