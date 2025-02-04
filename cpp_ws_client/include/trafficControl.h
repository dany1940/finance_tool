#ifndef ZMQ_HANDLER_H
#define ZMQ_HANDLER_H

#include <zmq.hpp> // ZeroMQ Library
#include <string> // Standard string library
#include <spdlog/spdlog.h> // Logging with spdlog
// Use standard namespace
using namespace std;
namespace zmqHandler = zmq;
namespace log = spdlog;

/**
 * @brief ZMQHandler class manages ZeroMQ message sending.
 */
class ZMQHandler {
public:
    /**
     * @brief Constructs a ZMQHandler with a given server address and port.
     * @param serverAddr The IP address or hostname of the server.
     * @param port The port number for the connection.
     */
    ZMQHandler(const string &serverAddr, int port);

    /**
     * @brief Destructor for ZMQHandler.
     * Ensures proper cleanup of ZeroMQ resources.
     */
    ~ZMQHandler();

    /**
     * @brief Sends a message through ZeroMQ.
     * @param message The string message to be sent.
     */
    void sendMessage(const string &message);

    /**
     * @brief Closes the ZeroMQ connection.
     * Ensures proper cleanup of the socket and context.
     */
    void closeConnection();

private:
    zmqHandler::context_t context; // ZeroMQ context for managing sockets
    zmqHandler::socket_t socket; // ZeroMQ socket for communication
};

#endif // ZMQ_HANDLER_H
