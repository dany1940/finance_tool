#include "zmqHandler.h"
#include <iostream>
#include <spdlog/spdlog.h> // ✅ Use spdlog for logging

using namespace std;
using namespace spdlog; // ✅ Use spdlog directly

// ✅ Constructor: Initializes ZeroMQ Server and binds to the specified endpoint
ZMQHandler::ZMQHandler(const string &server_addr, int port)
    : context(1), socket(context, ZMQ_PUSH) {
    string endpoint = "tcp://" + server_addr + ":" + to_string(port);
    socket.bind(endpoint);
    info("✅ ZeroMQ Server Bound to {}", endpoint);
}

// ✅ Sends a message via ZeroMQ
void ZMQHandler::sendMessage(const string &message) {
    zmq::message_t zmqMsg(message.begin(), message.end());
    socket.send(zmqMsg, zmq::send_flags::none);
    info("📤 Sent Message via ZeroMQ: {}", message);
}

// ✅ Closes the ZeroMQ connection
void ZMQHandler::closeConnection() {
    socket.close();
    context.close();
}

// ✅ Destructor: Ensures the connection is closed before destruction
ZMQHandler::~ZMQHandler() {
    closeConnection();
    info("🔌 ZeroMQ Connection Closed!");
    }
