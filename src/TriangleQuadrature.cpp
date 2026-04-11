#include "TriangleQuadrature.hpp"

namespace laplace {

TriangleQuadrature::TriangleQuadrature() {
  points_.push_back(QuadraturePoint{{{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}}, 0.5});
}

const std::vector<QuadraturePoint>& TriangleQuadrature::points() const {
  return points_;
}

}  // namespace laplace
