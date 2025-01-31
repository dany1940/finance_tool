#include "dpdk_handler.h"
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <iostream>

using namespace std;

void init_dpdk() {
    int argc = 2;
    char *argv[] = {"program", "--no-huge"};
    rte_eal_init(argc, argv);
    cout << "DPDK Initialized" << endl;
}
