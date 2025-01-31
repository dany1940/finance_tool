#include <iostream>
#include <thread>
#include "../include/websocket_client.h"
#include "../include/zmq_handler.h"
#include "../include/dpdk_handler.h"
#include "../include/zmq_kafka_producer.h"
#include "../include/traffic_control.h"

using namespace std;

int main() {
    // Initialize RDMA for Zero-Copy transmission
    ZMQHandler zmq_handler("127.0.0.1", 5555);

    std::string ws_url = "wss://streamer.finance.yahoo.com";
    WebSocketClient client(ws_url);

    client.connect_with_retry();  // ✅ Use retry function

    // ✅ Start heartbeat in a separate thread
    std::thread heartbeat_thread(&WebSocketClient::send_heartbeat, &client);

    while (true) {
        client.receive_message();
    }

    client.close();
    heartbeat_thread.join();  // ✅ Ensure heartbeat thread is closed properly
    return 0;

}
