#include <iostream>
#include <thread>
#include "websocket_client.h"
#include "rdma_handler.h"
#include "dpdk_handler.h"
#include "kafka_producer.h"
#include "traffic_control.h"

using namespace std;

int main() {
    // Initialize RDMA for Zero-Copy transmission
    init_rdma();

    // Initialize DPDK for high-speed networking
    init_dpdk();

    // Start WebSocket connections for multiple stock exchanges
    thread yahoo_thread(connect_to_exchange, "yahoo", "wss://streamer.finance.yahoo.com");

    yahoo_thread.join();

    return 0;
}
