#pragma once

#include "Types.hpp"

#include <vector>

namespace laplace {

struct ElementKernelOutput {
  std::vector<double> stiffness;
  std::vector<double> load;
};

ElementKernelOutput computeElementLocalSystemsKokkos(const MeshPartition& partition);

}  // namespace laplace
