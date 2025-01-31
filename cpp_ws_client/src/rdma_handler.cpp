#include "rdma_handler.h"
#include <rdma/rdma_cma.h>
#include <iostream>

using namespace std;

void init_rdma() {
    struct rdma_event_channel *ec;
    struct rdma_cm_id *id;
    ec = rdma_create_event_channel();
    rdma_create_id(ec, &id, NULL, RDMA_PS_TCP);
    cout << "RDMA Initialized" << endl;
}
