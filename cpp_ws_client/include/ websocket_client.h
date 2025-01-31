#ifndef WEBSOCKET_CLIENT_H
#define WEBSOCKET_CLIENT_H

#include <string>
#include <websocketpp/client.hpp>
#include <websocketpp/config/asio_no_tls_client.hpp>

using namespace std;
using client = websocketpp::client<websocketpp::config::asio_client>;

void connect_to_exchange(string exchange_name, string exchange_url);
void on_message(string exchange, websocketpp::connection_hdl hdl, client::message_ptr msg);
void send_heartbeat(const string &exchange);
void send_to_fastapi(const string &exchange, const string &payload);

#endif // WEBSOCKET_CLIENT_H
