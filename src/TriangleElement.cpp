#include "TriangleElement.hpp"

#include "ProblemData.hpp"

#include <stdexcept>

namespace laplace {

TriangleElement::TriangleElement(const TriangleQuadrature& quadrature)
    : quadrature_(quadrature) {}

LocalElementMatrix TriangleElement::computeLocalSystem(
  const Mesh& mesh,
  const Element& element) const {
  return computeLocalSystem({
    mesh.nodes[element.nodeIds[0]],
    mesh.nodes[element.nodeIds[1]],
    mesh.nodes[element.nodeIds[2]]
  });
}

LocalElementMatrix TriangleElement::computeLocalSystem(
  const std::array<Node, 3>& nodes) const {
  const Node& n0 = nodes[0];
  const Node& n1 = nodes[1];
  const Node& n2 = nodes[2];

  const double twiceArea =
    (n1.x - n0.x) * (n2.y - n0.y) - (n2.x - n0.x) * (n1.y - n0.y);
  const double area = 0.5 * twiceArea;
  if (area <= 0.0) {
    throw std::runtime_error("Encountered an element with non-positive area.");
  }

  const std::array<double, 3> b{
    n1.y - n2.y,
    n2.y - n0.y,
    n0.y - n1.y
  };
  const std::array<double, 3> c{
    n2.x - n1.x,
    n0.x - n2.x,
    n1.x - n0.x
  };

  LocalElementMatrix local;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      local.stiffness[i][j] = (b[i] * b[j] + c[i] * c[j]) / (4.0 * area);
    }
  }

  for (const QuadraturePoint& qp : quadrature_.points()) {
    const double x =
      qp.barycentric[0] * n0.x +
      qp.barycentric[1] * n1.x +
      qp.barycentric[2] * n2.x;
    const double y =
      qp.barycentric[0] * n0.y +
      qp.barycentric[1] * n1.y +
      qp.barycentric[2] * n2.y;
    const double fValue = forcingTerm(x, y);
    const double physicalWeight = qp.weight * 2.0 * area;
    for (int a = 0; a < 3; ++a) {
      local.load[a] += fValue * qp.barycentric[a] * physicalWeight;
    }
  }

  return local;
}

}  // namespace laplace
