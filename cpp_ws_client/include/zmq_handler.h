#ifndef ZMQ_HANDLER_H
#define ZMQ_HANDLER_H

#include <zmq.hpp>
#include <string>
#include <iostream>

class ZMQHandler {
public:
    ZMQHandler(const std::string &server_addr, int port);
    ~ZMQHandler();

    void send_message(const std::string &message);
    void close_connection();

private:
    zmq::context_t context;
    zmq::socket_t socket;
};

#endif // ZMQ_HANDLER_H
