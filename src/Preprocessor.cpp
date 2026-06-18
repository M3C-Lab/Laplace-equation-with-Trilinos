#include "Preprocessor.hpp"

#include "MeshGenerator.hpp"
#include "PartitionIO.hpp"

#include <metis.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace laplace {

namespace {

std::vector<std::vector<int>> buildNodeToElementAdjacency(const Mesh& mesh) {
  std::vector<std::vector<int>> nodeToElements(mesh.nodes.size());
  for (std::size_t elemId = 0; elemId < mesh.elements.size(); ++elemId) {
    const Element& element = mesh.elements[elemId];
    for (const int nodeId : element.nodeIds) {
      nodeToElements[static_cast<std::size_t>(nodeId)].push_back(static_cast<int>(elemId));
    }
  }
  return nodeToElements;
}

MeshPartition buildSerialPartition(const Mesh& mesh) {
  MeshPartition partition;
  partition.rank = 0;
  partition.size = 1;

  const std::size_t numNodes = mesh.nodes.size();
  const std::size_t numElements = mesh.elements.size();

  partition.ownedNodeIds.resize(numNodes);
  std::iota(partition.ownedNodeIds.begin(), partition.ownedNodeIds.end(), 0);
  partition.localToGlobalNodeIds = partition.ownedNodeIds;
  partition.localNodes = mesh.nodes;

  partition.localElementIds.resize(numElements);
  std::iota(partition.localElementIds.begin(), partition.localElementIds.end(), 0);
  partition.lien.reserve(numElements);
  for (const Element& element : mesh.elements) {
    partition.lien.push_back(element.nodeIds);
  }

  return partition;
}

MeshPartition buildPartitionForRank(
  const Mesh& mesh,
  const PartitionSummary& summary,
  const std::vector<std::vector<int>>& nodeToElements,
  const int rank) {
  MeshPartition partition;
  partition.rank = rank;
  partition.size = summary.cpuSize;

  for (std::size_t nodeId = 0; nodeId < summary.nodeOwners.size(); ++nodeId) {
    if (summary.nodeOwners[nodeId] == rank) {
      partition.ownedNodeIds.push_back(static_cast<int>(nodeId));
    }
  }

  std::vector<unsigned char> elementTouched(mesh.elements.size(), 0);
  for (const int nodeId : partition.ownedNodeIds) {
    for (const int elemId : nodeToElements[static_cast<std::size_t>(nodeId)]) {
      if (elementTouched[static_cast<std::size_t>(elemId)] == 0) {
        elementTouched[static_cast<std::size_t>(elemId)] = 1;
        partition.localElementIds.push_back(elemId);
      }
    }
  }

  std::vector<unsigned char> isOwnedNode(mesh.nodes.size(), 0);
  for (const int nodeId : partition.ownedNodeIds) {
    isOwnedNode[static_cast<std::size_t>(nodeId)] = 1;
  }

  std::vector<unsigned char> isRequiredGhost(mesh.nodes.size(), 0);
  for (const int elementId : partition.localElementIds) {
    const Element& element = mesh.elements[static_cast<std::size_t>(elementId)];
    for (const int nodeId : element.nodeIds) {
      if (isOwnedNode[static_cast<std::size_t>(nodeId)] == 0 &&
          isRequiredGhost[static_cast<std::size_t>(nodeId)] == 0) {
        isRequiredGhost[static_cast<std::size_t>(nodeId)] = 1;
        partition.ghostNodeIds.push_back(nodeId);
      }
    }
  }

  std::sort(partition.ghostNodeIds.begin(), partition.ghostNodeIds.end());
  partition.localToGlobalNodeIds = partition.ownedNodeIds;
  partition.localToGlobalNodeIds.insert(
    partition.localToGlobalNodeIds.end(),
    partition.ghostNodeIds.begin(),
    partition.ghostNodeIds.end());

  partition.localNodes.reserve(partition.localToGlobalNodeIds.size());
  std::vector<int> globalToLocalNode(mesh.nodes.size(), -1);
  for (std::size_t localNodeId = 0; localNodeId < partition.localToGlobalNodeIds.size(); ++localNodeId) {
    const int globalNodeId = partition.localToGlobalNodeIds[localNodeId];
    partition.localNodes.push_back(mesh.nodes[static_cast<std::size_t>(globalNodeId)]);
    globalToLocalNode[static_cast<std::size_t>(globalNodeId)] = static_cast<int>(localNodeId);
  }

  partition.lien.reserve(partition.localElementIds.size());
  for (const int elementId : partition.localElementIds) {
    const Element& element = mesh.elements[static_cast<std::size_t>(elementId)];
    std::array<int, 3> localRow{};
    for (int i = 0; i < 3; ++i) {
      const int localNodeId = globalToLocalNode[static_cast<std::size_t>(element.nodeIds[i])];
      if (localNodeId < 0) {
        throw std::runtime_error("Failed to generate LIEN for local partition.");
      }
      localRow[static_cast<std::size_t>(i)] = localNodeId;
    }
    partition.lien.push_back(localRow);
  }

  return partition;
}

}  // namespace

std::string buildPartitionFileStem(const int rank) {
  return "partition.rank" + std::to_string(rank);
}

PartitionSummary partitionMeshWithMetis(
  const Mesh& mesh,
  const int partitions,
  const int metisNCommon,
  const bool useDualGraph) {
  PartitionSummary summary;
  summary.cpuSize = partitions;
  summary.elementOwners.resize(mesh.elements.size(), 0);
  summary.nodeOwners.resize(mesh.nodes.size(), 0);

  if (partitions <= 1) {
    return summary;
  }

  idx_t ne = static_cast<idx_t>(mesh.elements.size());
  idx_t nn = static_cast<idx_t>(mesh.nodes.size());
  idx_t ncommon = static_cast<idx_t>(metisNCommon);
  idx_t nparts = static_cast<idx_t>(partitions);
  idx_t objval = 0;

  std::vector<idx_t> eptr(static_cast<std::size_t>(ne) + 1, 0);
  std::vector<idx_t> eind(static_cast<std::size_t>(ne) * 3, 0);

  for (idx_t e = 0; e < ne; ++e) {
    eptr[static_cast<std::size_t>(e)] = 3 * e;
    const auto& element = mesh.elements[static_cast<std::size_t>(e)];
    for (int i = 0; i < 3; ++i) {
      eind[static_cast<std::size_t>(3 * e + i)] = static_cast<idx_t>(element.nodeIds[i]);
    }
  }
  eptr[static_cast<std::size_t>(ne)] = 3 * ne;

  std::vector<idx_t> epart(static_cast<std::size_t>(ne), 0);
  std::vector<idx_t> npart(static_cast<std::size_t>(nn), 0);

  int status = METIS_OK;
  if (useDualGraph) {
    status = METIS_PartMeshDual(
      &ne,
      &nn,
      eptr.data(),
      eind.data(),
      nullptr,
      nullptr,
      &ncommon,
      &nparts,
      nullptr,
      nullptr,
      &objval,
      epart.data(),
      npart.data());
  } else {
    status = METIS_PartMeshNodal(
      &ne,
      &nn,
      eptr.data(),
      eind.data(),
      nullptr,
      nullptr,
      &nparts,
      nullptr,
      nullptr,
      &objval,
      epart.data(),
      npart.data());
  }

  if (status != METIS_OK) {
    throw std::runtime_error("METIS partitioning failed.");
  }

  for (std::size_t i = 0; i < summary.elementOwners.size(); ++i) {
    summary.elementOwners[i] = static_cast<int>(epart[i]);
  }
  for (std::size_t i = 0; i < summary.nodeOwners.size(); ++i) {
    summary.nodeOwners[i] = static_cast<int>(npart[i]);
  }

  return summary;
}

std::vector<MeshPartition> buildMeshPartitions(
  const Mesh& mesh,
  const PartitionSummary& summary) {
  if (summary.cpuSize == 1) {
    return {buildSerialPartition(mesh)};
  }

  const auto nodeToElements = buildNodeToElementAdjacency(mesh);
  std::vector<MeshPartition> partitions;
  partitions.reserve(static_cast<std::size_t>(summary.cpuSize));
  for (int rank = 0; rank < summary.cpuSize; ++rank) {
    partitions.push_back(buildPartitionForRank(mesh, summary, nodeToElements, rank));
  }
  return partitions;
}

void runPreprocessor(const PreprocessorOptions& options) {
  const Mesh mesh = buildStructuredTriangularMesh(options.nx, options.ny);
  const MeshMetadata metadata{
    .nx = options.nx,
    .ny = options.ny,
    .numNodes = static_cast<int>(mesh.nodes.size()),
    .numElements = static_cast<int>(mesh.elements.size())
  };

  const PartitionSummary summary = partitionMeshWithMetis(
    mesh,
    options.partitions,
    options.metisNCommon,
    options.useDualGraph);
  const auto partitions = buildMeshPartitions(mesh, summary);

  writePreprocessedMesh(mesh, metadata, options.outputDirectory);
  for (const auto& partition : partitions) {
    writeMeshPartition(partition, options.outputDirectory);
  }

  std::cout << "Preprocessed mesh written to " << options.outputDirectory << '\n';
  std::cout << "  nodes    = " << mesh.nodes.size() << '\n';
  std::cout << "  elements = " << mesh.elements.size() << '\n';
  std::cout << "  parts    = " << options.partitions << '\n';
}

}  // namespace laplace
