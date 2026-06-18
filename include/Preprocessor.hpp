#pragma once

#include "Types.hpp"

#include <string>
#include <vector>

namespace laplace {

PartitionSummary partitionMeshWithMetis(
  const Mesh& mesh,
  int partitions,
  int metisNCommon,
  bool useDualGraph);

std::vector<MeshPartition> buildMeshPartitions(
  const Mesh& mesh,
  const PartitionSummary& summary);

void runPreprocessor(const PreprocessorOptions& options);

std::string buildPartitionFileStem(int rank);

}  // namespace laplace
