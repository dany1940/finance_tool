#include "websocket_client.h"
#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>
#include <iostream>
#include <json/json.h>
#include "kafka_producer.h"
#include "traffic_control.h"

using namespace std;
using namespace Json;

unordered_map<string, int> message_retries;

void send_heartbeat(const string &exchange) {
    cout << "Heartbeat sent to " << exchange << endl;
}

void send_to_fastapi(const string &exchange, const string &payload) {
    client c;
    websocketpp::lib::error_code ec;
    string uri = "ws://localhost:8000/ws";
    auto con = c.get_connection(uri, ec);

    if (ec) {
        cerr << "Error connecting to FastAPI: " << ec.message() << endl;
        return;
    }

    c.send(con->get_handle(), payload, websocketpp::frame::opcode::text);
    cout << "[" << exchange << "] Sent data to FastAPI WebSocket: " << payload << endl;
}

void on_message(string exchange, websocketpp::connection_hdl hdl, client::message_ptr msg) {
    string payload = msg->get_payload();
    Json::CharReaderBuilder reader;
    Json::Value root;
    string errors;
    istringstream s(payload);

    if (Json::parseFromStream(reader, s, &root, &errors)) {
        string message_id = root["message_id"].asString();
        string symbol = root["symbol"].asString();
        string price = root["price"].asString();

        cout << "[" << exchange << "] Received: " << symbol << " - $" << price << endl;

        send_to_fastapi(exchange, payload);
        send_heartbeat(exchange);

        if (message_retries[message_id] < 3) {
            produce_to_kafka("stock_data", payload);
            message_retries[message_id]++;
        } else {
            cout << "Message " << message_id << " failed. Requesting resend..." << endl;
            system(("curl -X POST http://localhost:8000/resend-missing-message -d '{\"message_id\": \"" + message_id + "\"}'").c_str());
        }
    }
}

void connect_to_exchange(string exchange_name, string exchange_url) {
    client c;
    c.init_asio();

    websocketpp::lib::error_code ec;
    auto con = c.get_connection(exchange_url, ec);

    if (ec) {
        cerr << "WebSocket Connection Failed for " << exchange_name << ": " << ec.message() << endl;
        return;
    }

    c.set_message_handler(bind(&on_message, exchange_name, placeholders::_1, placeholders::_2));
    c.connect(con);
    c.run();
}
