#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace std;

// Lookup Table for Normal CDF
extern vector<double> normalCDF_LUT;

// Utility functions
void initializeLookupTable();
double normalCDF(double x);
double normalPDF(double x);
void logMessage(const string& message);

#endif // UTILS_H
