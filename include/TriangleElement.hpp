#pragma once

#include "TriangleQuadrature.hpp"
#include "Types.hpp"

#include <array>

namespace laplace {

struct LocalElementMatrix {
  std::array<std::array<double, 3>, 3> stiffness{};
  std::array<double, 3> load{};
};

class TriangleElement {
 public:
  explicit TriangleElement(const TriangleQuadrature& quadrature);

  LocalElementMatrix computeLocalSystem(
    const Mesh& mesh,
    const Element& element) const;

  LocalElementMatrix computeLocalSystem(
    const std::array<Node, 3>& nodes) const;

 private:
  const TriangleQuadrature& quadrature_;
};

}  // namespace laplace
