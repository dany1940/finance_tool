#ifndef ZMQ_HANDLER_H
#define ZMQ_HANDLER_H

#include <zmq.hpp>
#include <string>
#include <iostream>

using namespace std;
using namespace zmq;



class ZMQHandler {
public:
    ZMQHandler(const string &server_addr, int port);
    ~ZMQHandler();

    void sendMessage(const string &message);
    void closeConnection();

private:
    context_t context;
    socket_t socket;
};

#endif // ZMQ_HANDLER_H
