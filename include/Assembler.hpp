#pragma once

#include "TriangleElement.hpp"
#include "Types.hpp"

namespace laplace {

class Assembler {
 public:
  explicit Assembler(const TriangleElement& element);

  DiscreteSystem assemble(const Mesh& mesh, const MeshPartition& partition) const;
  int lastAssemblyThreadCount() const;

 private:
  const TriangleElement& element_;
  mutable int lastAssemblyThreadCount_ = 1;
};

}  // namespace laplace
