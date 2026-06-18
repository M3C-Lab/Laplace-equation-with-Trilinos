#pragma once

#include <array>
#include <string>
#include <map>
#include <vector>

namespace laplace {

struct Node {
  double x = 0.0;
  double y = 0.0;
  bool isBoundary = false;
};

struct Element {
  std::array<int, 3> nodeIds{};
};

struct Mesh {
  int nx = 8;
  int ny = 8;
  std::vector<Node> nodes;
  std::vector<Element> elements;
};

struct MeshPartition {
  int rank = 0;
  int size = 1;
  std::vector<Node> localNodes;
  std::vector<int> ownedNodeIds;
  std::vector<int> ghostNodeIds;
  std::vector<int> localToGlobalNodeIds;
  std::vector<int> localElementIds;
  std::vector<std::array<int, 3>> lien;
};

struct PartitionSummary {
  int cpuSize = 1;
  std::vector<int> elementOwners;
  std::vector<int> nodeOwners;
};

struct MeshMetadata {
  int nx = 0;
  int ny = 0;
  int numNodes = 0;
  int numElements = 0;
};

struct PreprocessedMesh {
  Mesh mesh;
  MeshMetadata metadata;
};

struct PreprocessorOptions {
  int nx = 16;
  int ny = 16;
  int partitions = 1;
  int metisNCommon = 2;
  bool useDualGraph = true;
  std::string outputDirectory;
};

struct DiscreteSystem {
  std::vector<int> nodeToDof;
  std::vector<long long> ownedGlobalDofs;
  std::vector<int> ownedNodeIds;
  std::vector<int> globalDofToLocalRow;
  std::vector<std::map<long long, double>> stiffnessRows;
  std::vector<double> rhs;
};

struct PostProcessSummary {
  std::vector<double> exactValues;
  std::vector<double> errorValues;
  double maxAbsError = 0.0;
  double l2Error = 0.0;
};

}  // namespace laplace
