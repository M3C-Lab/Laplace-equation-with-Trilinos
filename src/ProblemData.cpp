#include "ProblemData.hpp"

#include <cmath>

namespace laplace {

namespace detail {

double exactSolutionDevice(const double x, const double y) {
  return std::sin(kPi * x) * std::sin(kPi * y);
}

}  // namespace detail

double exactSolution(const double x, const double y) {
  return detail::exactSolutionDevice(x, y);
}

double forcingTerm(const double x, const double y) {
  return 2.0 * kPi * kPi * detail::exactSolutionDevice(x, y);
}

}  // namespace laplace
