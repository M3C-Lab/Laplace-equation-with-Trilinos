#include "PartitionIO.hpp"

#include "HDF5_IO.hpp"
#include "Preprocessor.hpp"

#include <filesystem>
#include <stdexcept>

namespace laplace {

namespace {

std::filesystem::path meshFilePath(const std::string& directory) {
  return std::filesystem::path(directory) / "mesh.h5";
}

std::filesystem::path partitionFilePath(const std::string& directory, const int rank) {
  return std::filesystem::path(directory) / (buildPartitionFileStem(rank) + ".h5");
}

std::vector<double> flattenNodes(const std::vector<Node>& nodes) {
  std::vector<double> values;
  values.reserve(nodes.size() * 3);
  for (const Node& node : nodes) {
    values.push_back(node.x);
    values.push_back(node.y);
    values.push_back(node.isBoundary ? 1.0 : 0.0);
  }
  return values;
}

std::vector<int> flattenElements(const std::vector<Element>& elements) {
  std::vector<int> values;
  values.reserve(elements.size() * 3);
  for (const Element& element : elements) {
    values.push_back(element.nodeIds[0]);
    values.push_back(element.nodeIds[1]);
    values.push_back(element.nodeIds[2]);
  }
  return values;
}

std::vector<int> flattenLien(const std::vector<std::array<int, 3>>& lien) {
  std::vector<int> values;
  values.reserve(lien.size() * 3);
  for (const auto& row : lien) {
    values.push_back(row[0]);
    values.push_back(row[1]);
    values.push_back(row[2]);
  }
  return values;
}

std::vector<Node> unflattenNodes(const std::vector<double>& values) {
  if (values.size() % 3 != 0) {
    throw std::runtime_error("Invalid HDF5 node array length.");
  }

  std::vector<Node> nodes(values.size() / 3);
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    Node node;
    node.x = values[3 * i + 0];
    node.y = values[3 * i + 1];
    node.isBoundary = (values[3 * i + 2] != 0.0);
    nodes[i] = node;
  }
  return nodes;
}

std::vector<Element> unflattenElements(const std::vector<int>& values) {
  if (values.size() % 3 != 0) {
    throw std::runtime_error("Invalid HDF5 element array length.");
  }

  std::vector<Element> elements(values.size() / 3);
  for (std::size_t i = 0; i < elements.size(); ++i) {
    elements[i].nodeIds = {
      values[3 * i + 0],
      values[3 * i + 1],
      values[3 * i + 2]
    };
  }
  return elements;
}

std::vector<std::array<int, 3>> unflattenLien(const std::vector<int>& values) {
  if (values.size() % 3 != 0) {
    throw std::runtime_error("Invalid HDF5 LIEN array length.");
  }

  std::vector<std::array<int, 3>> lien(values.size() / 3);
  for (std::size_t i = 0; i < lien.size(); ++i) {
    lien[i] = {
      values[3 * i + 0],
      values[3 * i + 1],
      values[3 * i + 2]
    };
  }
  return lien;
}

}  // namespace

PreprocessedMesh readPreprocessedMesh(const std::string& directory) {
  const HDF5Reader reader(meshFilePath(directory).string());
  PreprocessedMesh result;

  result.metadata.nx = reader.readIntScalar("/metadata", "nx");
  result.metadata.ny = reader.readIntScalar("/metadata", "ny");
  result.metadata.numNodes = reader.readIntScalar("/metadata", "num_nodes");
  result.metadata.numElements = reader.readIntScalar("/metadata", "num_elements");

  result.mesh.nx = result.metadata.nx;
  result.mesh.ny = result.metadata.ny;
  result.mesh.nodes = unflattenNodes(reader.readDoubleVector("/mesh", "nodes"));
  result.mesh.elements = unflattenElements(reader.readIntVector("/mesh", "elements"));

  return result;
}

MeshPartition readMeshPartition(const std::string& directory, const int rank) {
  const HDF5Reader reader(partitionFilePath(directory, rank).string());
  MeshPartition partition;

  partition.rank = reader.readIntScalar("/partition", "rank");
  partition.size = reader.readIntScalar("/partition", "cpu_size");
  partition.localNodes = unflattenNodes(reader.readDoubleVector("/partition", "local_nodes"));
  partition.ownedNodeIds = reader.readIntVector("/partition", "owned_nodes");
  partition.ghostNodeIds = reader.readIntVector("/partition", "ghost_nodes");
  partition.localToGlobalNodeIds = reader.readIntVector("/partition", "local_to_global");
  partition.localElementIds = reader.readIntVector("/partition", "local_elements");
  partition.lien = unflattenLien(reader.readIntVector("/partition", "lien"));

  return partition;
}

void writePreprocessedMesh(
  const Mesh& mesh,
  const MeshMetadata& metadata,
  const std::string& directory) {
  std::filesystem::create_directories(directory);

  HDF5Writer writer(meshFilePath(directory).string());
  const hid_t metadataGroup = writer.createGroup("/metadata");
  writer.writeIntScalar(metadataGroup, "nx", metadata.nx);
  writer.writeIntScalar(metadataGroup, "ny", metadata.ny);
  writer.writeIntScalar(metadataGroup, "num_nodes", metadata.numNodes);
  writer.writeIntScalar(metadataGroup, "num_elements", metadata.numElements);
  writer.closeGroup(metadataGroup);

  const hid_t meshGroup = writer.createGroup("/mesh");
  writer.writeDoubleVector(meshGroup, "nodes", flattenNodes(mesh.nodes));
  writer.writeIntVector(meshGroup, "elements", flattenElements(mesh.elements));
  writer.closeGroup(meshGroup);
}

void writeMeshPartition(const MeshPartition& partition, const std::string& directory) {
  std::filesystem::create_directories(directory);

  HDF5Writer writer(partitionFilePath(directory, partition.rank).string());
  const hid_t partitionGroup = writer.createGroup("/partition");
  writer.writeIntScalar(partitionGroup, "rank", partition.rank);
  writer.writeIntScalar(partitionGroup, "cpu_size", partition.size);
  writer.writeDoubleVector(partitionGroup, "local_nodes", flattenNodes(partition.localNodes));
  writer.writeIntVector(partitionGroup, "owned_nodes", partition.ownedNodeIds);
  writer.writeIntVector(partitionGroup, "ghost_nodes", partition.ghostNodeIds);
  writer.writeIntVector(partitionGroup, "local_to_global", partition.localToGlobalNodeIds);
  writer.writeIntVector(partitionGroup, "local_elements", partition.localElementIds);
  writer.writeIntVector(partitionGroup, "lien", flattenLien(partition.lien));
  writer.closeGroup(partitionGroup);
}

}  // namespace laplace
