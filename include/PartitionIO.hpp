#pragma once

#include "Types.hpp"

#include <string>

namespace laplace {

PreprocessedMesh readPreprocessedMesh(const std::string& directory);
MeshPartition readMeshPartition(const std::string& directory, int rank);

void writePreprocessedMesh(
  const Mesh& mesh,
  const MeshMetadata& metadata,
  const std::string& directory);

void writeMeshPartition(
  const MeshPartition& partition,
  const std::string& directory);

}  // namespace laplace
