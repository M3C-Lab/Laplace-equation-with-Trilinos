#pragma once

namespace laplace {

constexpr double kPi = 3.14159265358979323846;

double exactSolution(double x, double y);
double forcingTerm(double x, double y);

}  // namespace laplace
