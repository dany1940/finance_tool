#ifndef BLACK_SCHOLES_H
#define BLACK_SCHOLES_H

double blackScholes(double S, double K, double T, double r, double sigma, bool isCall);
double normalCDF(double x);

#endif // BLACK_SCHOLES_H
