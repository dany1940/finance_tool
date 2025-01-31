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

    // Initialize DPDK for high-speed networking

    // Start WebSocket connections for multiple stock exchanges
    thread yahoo_thread(connect_to_exchange, "yahoo", "wss://streamer.finance.yahoo.com");

    yahoo_thread.join();

    return 0;
}
