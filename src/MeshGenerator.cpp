#include "MeshGenerator.hpp"

#include <stdexcept>

namespace laplace {

Mesh buildStructuredTriangularMesh(const int nx, const int ny) {
  if (nx < 1 || ny < 1) {
    throw std::invalid_argument("nx and ny must both be positive.");
  }

  Mesh mesh;
  mesh.nx = nx;
  mesh.ny = ny;

  const double hx = 1.0 / static_cast<double>(nx);
  const double hy = 1.0 / static_cast<double>(ny);

  mesh.nodes.reserve(static_cast<std::size_t>((nx + 1) * (ny + 1)));
  for (int j = 0; j <= ny; ++j) {
    for (int i = 0; i <= nx; ++i) {
      Node node;
      node.x = static_cast<double>(i) * hx;
      node.y = static_cast<double>(j) * hy;
      node.isBoundary = (i == 0 || i == nx || j == 0 || j == ny);
      mesh.nodes.push_back(node);
    }
  }

  const auto nodeIndex = [nx](const int i, const int j) {
    return j * (nx + 1) + i;
  };

  mesh.elements.reserve(static_cast<std::size_t>(2 * nx * ny));
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const int n0 = nodeIndex(i, j);
      const int n1 = nodeIndex(i + 1, j);
      const int n2 = nodeIndex(i, j + 1);
      const int n3 = nodeIndex(i + 1, j + 1);

      mesh.elements.push_back(Element{{n0, n1, n3}});
      mesh.elements.push_back(Element{{n0, n3, n2}});
    }
  }

  return mesh;
}

}  // namespace laplace
