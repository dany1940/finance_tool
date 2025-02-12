#include "utils.h"

vector<double> normalCDF_LUT(1000, 0.0);

void initializeLookupTable() {
    #pragma omp parallel for
    for (size_t i = 0; i < normalCDF_LUT.size(); i++) {
        double x = -5.0 + 10.0 * (double(i) / normalCDF_LUT.size());
        normalCDF_LUT[i] = 0.5 * (1 + erf(x / sqrt(2.0)));
    }
    logMessage("Lookup table for normal CDF initialized.");
}

double normalCDF(double x) {
    size_t index = (size_t)((x + 5.0) / 10.0 * normalCDF_LUT.size());
    if (index >= normalCDF_LUT.size()) return 1.0;
    if (index < 0) return 0.0;
    return normalCDF_LUT[index];
}

double normalPDF(double x) {
    return exp(-0.5 * x * x) / sqrt(2.0 * M_PI);
}

void logMessage(const string& message) {
    auto now = chrono::system_clock::to_time_t(chrono::system_clock::now());
    cout << "[LOG] " << put_time(localtime(&now), "%Y-%m-%d %H:%M:%S") << " " << message << endl;
}
