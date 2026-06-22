#include "ElementKernel.hpp"

#include "ProblemData.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>

namespace laplace {

namespace {

KOKKOS_INLINE_FUNCTION
double exactSolutionKernel(const double x, const double y) {
  return sin(kPi * x) * sin(kPi * y);
}

KOKKOS_INLINE_FUNCTION
double forcingTermKernel(const double x, const double y) {
  return 2.0 * kPi * kPi * exactSolutionKernel(x, y);
}

KOKKOS_INLINE_FUNCTION
void computeLocalSystemKernel(
  const double x0,
  const double y0,
  const double x1,
  const double y1,
  const double x2,
  const double y2,
  double stiffness[3][3],
  double load[3]) {
  const double twiceArea = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
  const double area = 0.5 * twiceArea;

  const double b[3] = {y1 - y2, y2 - y0, y0 - y1};
  const double c[3] = {x2 - x1, x0 - x2, x1 - x0};
  const double x = (x0 + x1 + x2) / 3.0;
  const double y = (y0 + y1 + y2) / 3.0;
  const double fValue = forcingTermKernel(x, y);

  for (int i = 0; i < 3; ++i) {
    load[i] = fValue * (area / 3.0);
    for (int j = 0; j < 3; ++j) {
      stiffness[i][j] = (b[i] * b[j] + c[i] * c[j]) / (4.0 * area);
    }
  }
}

}  // namespace

ElementKernelOutput computeElementLocalSystemsKokkos(const MeshPartition& partition) {
  using execution_space = Kokkos::DefaultExecutionSpace;
  const std::size_t localElementCount = partition.localElementIds.size();

  ElementKernelOutput output;
  output.stiffness.resize(localElementCount * 9, 0.0);
  output.load.resize(localElementCount * 3, 0.0);

  Kokkos::View<double*, execution_space> elementStiffness("element_stiffness", localElementCount * 9);
  Kokkos::View<double*, execution_space> elementLoad("element_load", localElementCount * 3);
  Kokkos::View<double*, execution_space> elementCoords("element_coords", localElementCount * 6);

  auto coordsHost = Kokkos::create_mirror_view(elementCoords);
  for (std::size_t localElemIndex = 0; localElemIndex < localElementCount; ++localElemIndex) {
    const auto& lien = partition.lien[localElemIndex];
    for (int a = 0; a < 3; ++a) {
      const Node& node = partition.localNodes[static_cast<std::size_t>(lien[a])];
      coordsHost(localElemIndex * 6 + 2 * a) = node.x;
      coordsHost(localElemIndex * 6 + 2 * a + 1) = node.y;
    }
  }
  Kokkos::deep_copy(elementCoords, coordsHost);

  Kokkos::parallel_for(
    "assemble_local_elements",
    Kokkos::RangePolicy<execution_space>(0, static_cast<int>(localElementCount)),
    KOKKOS_LAMBDA(const int elemIndex) {
      double stiffness[3][3];
      double load[3];
      const std::size_t coordBase = static_cast<std::size_t>(elemIndex) * 6;
      const std::size_t loadBase = static_cast<std::size_t>(elemIndex) * 3;
      const std::size_t stiffnessBase = static_cast<std::size_t>(elemIndex) * 9;
      computeLocalSystemKernel(
        elementCoords(coordBase),
        elementCoords(coordBase + 1),
        elementCoords(coordBase + 2),
        elementCoords(coordBase + 3),
        elementCoords(coordBase + 4),
        elementCoords(coordBase + 5),
        stiffness,
        load);
      for (int a = 0; a < 3; ++a) {
        elementLoad(loadBase + static_cast<std::size_t>(a)) = load[a];
        for (int b = 0; b < 3; ++b) {
          elementStiffness(stiffnessBase + static_cast<std::size_t>(a * 3 + b)) = stiffness[a][b];
        }
      }
    });
  execution_space().fence();

  auto stiffnessHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), elementStiffness);
  auto loadHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), elementLoad);

  for (std::size_t i = 0; i < output.stiffness.size(); ++i) {
    output.stiffness[i] = stiffnessHost(i);
  }
  for (std::size_t i = 0; i < output.load.size(); ++i) {
    output.load[i] = loadHost(i);
  }

  return output;
}

}  // namespace laplace
