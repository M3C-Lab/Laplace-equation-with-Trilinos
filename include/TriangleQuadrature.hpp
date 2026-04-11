#pragma once

#include <array>
#include <vector>

namespace laplace {

struct QuadraturePoint {
  std::array<double, 3> barycentric{};
  double weight = 0.0;
};

class TriangleQuadrature {
 public:
  TriangleQuadrature();

  const std::vector<QuadraturePoint>& points() const;

 private:
  std::vector<QuadraturePoint> points_;
};

}  // namespace laplace
