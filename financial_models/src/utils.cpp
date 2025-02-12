#include "utils.h"
#include <iostream>
#include <cmath>

using namespace std;

void logMessage(const string& message) {
    cerr << "[LOG] " << message << endl;
}

double normalCDF(double x) {
    return 0.5 * erfc(-x * M_SQRT1_2);
}

double normalPDF(double x) {
    return exp(-0.5 * x * x) / sqrt(2.0 * M_PI);
}
