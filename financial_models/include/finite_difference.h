#ifndef FINITE_DIFFERENCE_H
#define FINITE_DIFFERENCE_H

double finiteDifferenceBlackScholes(double S, double K, double T, double r, double sigma, bool isCall, int gridSize, int timeSteps);

#endif // FINITE_DIFFERENCE_H
