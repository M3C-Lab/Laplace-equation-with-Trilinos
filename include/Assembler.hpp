#pragma once

#include "TriangleElement.hpp"
#include "Types.hpp"

namespace laplace {

MeshPartition buildNodeRowPartition(
  const Mesh& mesh,
  int mpiRank,
  int mpiSize,
  bool useMpiPartition);

class Assembler {
 public:
  explicit Assembler(const TriangleElement& element);

  DiscreteSystem assemble(const Mesh& mesh, const MeshPartition& partition) const;

 private:
  const TriangleElement& element_;
};

}  // namespace laplace
