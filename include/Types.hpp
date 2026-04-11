#pragma once

#include <array>
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
  bool useMpiPartition = false;
  int rank = 0;
  int size = 1;
  int ownedNodeRowBegin = 1;
  int ownedNodeRowEnd = 0;
  std::vector<int> localElementIds;
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
