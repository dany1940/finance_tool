#include "zmq_handler.h"
#include <iostream>

ZMQHandler::ZMQHandler(const std::string &server_addr, int port)
    : context(1), socket(context, ZMQ_PUSH) {
    std::string endpoint = "tcp://" + server_addr + ":" + std::to_string(port);
    socket.bind(endpoint);
    std::cout << "✅ ZeroMQ Server Bound to " << endpoint << std::endl;
}

void ZMQHandler::send_message(const std::string &message) {
    zmq::message_t zmq_msg(message.begin(), message.end());
    socket.send(zmq_msg, zmq::send_flags::none);
    std::cout << "📤 Sent Message via ZeroMQ: " << message << std::endl;
}

void ZMQHandler::close_connection() {
    socket.close();
    context.close();
}

ZMQHandler::~ZMQHandler() {
    close_connection();
}
