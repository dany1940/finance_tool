#ifndef WEBSOCKET_CLIENT_H
#define WEBSOCKET_CLIENT_H

#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace boost::beast;
using namespace boost::asio;
using namespace boost::asio::ip;

// Function Declarations
void connect_to_exchange(const string &exchange_name, const string &exchange_url);
void send_heartbeat(const string &exchange);
void send_to_fastapi(const string &exchange, const string &payload);

#endif // WEBSOCKET_CLIENT_H
