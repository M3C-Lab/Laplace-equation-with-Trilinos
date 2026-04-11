#include "Assembler.hpp"

#include "ProblemData.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace laplace {

MeshPartition buildNodeRowPartition(
  const Mesh& mesh,
  const int mpiRank,
  const int mpiSize,
  const bool useMpiPartition) {
  MeshPartition partition;
  partition.useMpiPartition = useMpiPartition && mpiSize > 1;
  partition.rank = mpiRank;
  partition.size = mpiSize;

  const int interiorRows = std::max(0, mesh.ny - 1);
  if (!partition.useMpiPartition || interiorRows == 0) {
    partition.ownedNodeRowBegin = 1;
    partition.ownedNodeRowEnd = mesh.ny - 1;
    partition.localElementIds.resize(mesh.elements.size());
    for (std::size_t i = 0; i < mesh.elements.size(); ++i) {
      partition.localElementIds[i] = static_cast<int>(i);
    }
    return partition;
  }

  const int baseRows = interiorRows / mpiSize;
  const int extraRows = interiorRows % mpiSize;
  const int ownedRows = baseRows + (mpiRank < extraRows ? 1 : 0);
  const int prefixRows =
    mpiRank * baseRows + std::min(mpiRank, extraRows);

  if (ownedRows == 0) {
    partition.ownedNodeRowBegin = 1;
    partition.ownedNodeRowEnd = 0;
    return partition;
  }

  partition.ownedNodeRowBegin = 1 + prefixRows;
  partition.ownedNodeRowEnd = partition.ownedNodeRowBegin + ownedRows - 1;

  const int elementRowBegin = std::max(0, partition.ownedNodeRowBegin - 1);
  const int elementRowEnd = std::min(mesh.ny - 1, partition.ownedNodeRowEnd);
  const int elementsPerRow = 2 * mesh.nx;
  partition.localElementIds.reserve(
    static_cast<std::size_t>((elementRowEnd - elementRowBegin + 1) * elementsPerRow));

  for (int elemRow = elementRowBegin; elemRow <= elementRowEnd; ++elemRow) {
    const int begin = elemRow * elementsPerRow;
    const int end = begin + elementsPerRow;
    for (int elemId = begin; elemId < end; ++elemId) {
      partition.localElementIds.push_back(elemId);
    }
  }

  return partition;
}

Assembler::Assembler(const TriangleElement& element)
    : element_(element) {}

DiscreteSystem Assembler::assemble(const Mesh& mesh, const MeshPartition& partition) const {
  DiscreteSystem system;
  system.nodeToDof.assign(mesh.nodes.size(), -1);

  int dofCount = 0;
  for (std::size_t nodeId = 0; nodeId < mesh.nodes.size(); ++nodeId) {
    if (!mesh.nodes[nodeId].isBoundary) {
      system.nodeToDof[nodeId] = dofCount++;
    }
  }

  system.globalDofToLocalRow.assign(static_cast<std::size_t>(dofCount), -1);

  for (std::size_t nodeId = 0; nodeId < mesh.nodes.size(); ++nodeId) {
    const int dof = system.nodeToDof[nodeId];
    if (dof < 0) {
      continue;
    }

    const int nodeRow = static_cast<int>(nodeId) / (mesh.nx + 1);
    const bool ownedByRank =
      !partition.useMpiPartition ||
      (nodeRow >= partition.ownedNodeRowBegin && nodeRow <= partition.ownedNodeRowEnd);
    if (ownedByRank) {
      system.globalDofToLocalRow[static_cast<std::size_t>(dof)] =
        static_cast<int>(system.ownedGlobalDofs.size());
      system.ownedGlobalDofs.push_back(static_cast<long long>(dof));
      system.ownedNodeIds.push_back(static_cast<int>(nodeId));
    }
  }

  system.stiffnessRows.resize(system.ownedGlobalDofs.size());
  system.rhs.assign(system.ownedGlobalDofs.size(), 0.0);

#if defined(_OPENMP)
  std::vector<omp_lock_t> rowLocks(system.ownedGlobalDofs.size());
  for (std::size_t row = 0; row < rowLocks.size(); ++row) {
    omp_init_lock(&rowLocks[row]);
  }
#endif

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t elem = 0; elem < static_cast<std::ptrdiff_t>(partition.localElementIds.size()); ++elem) {
    const int elementId = partition.localElementIds[static_cast<std::size_t>(elem)];
    const Element& element = mesh.elements[static_cast<std::size_t>(elementId)];
    const auto local = element_.computeLocalSystem(mesh, element);

    for (int a = 0; a < 3; ++a) {
      const int globalNodeA = element.nodeIds[a];
      const int dofA = system.nodeToDof[globalNodeA];
      if (dofA < 0) {
        continue;
      }

      const int localRow = system.globalDofToLocalRow[static_cast<std::size_t>(dofA)];
      if (localRow < 0) {
        continue;
      }

      omp_set_lock(&rowLocks[static_cast<std::size_t>(localRow)]);
      system.rhs[static_cast<std::size_t>(localRow)] += local.load[a];

      for (int b = 0; b < 3; ++b) {
        const int globalNodeB = element.nodeIds[b];
        const int dofB = system.nodeToDof[globalNodeB];
        if (dofB >= 0) {
          system.stiffnessRows[static_cast<std::size_t>(localRow)][dofB] += local.stiffness[a][b];
        } else {
          const Node& boundaryNode = mesh.nodes[globalNodeB];
          system.rhs[static_cast<std::size_t>(localRow)] -=
            local.stiffness[a][b] * exactSolution(boundaryNode.x, boundaryNode.y);
        }
      }

      omp_unset_lock(&rowLocks[static_cast<std::size_t>(localRow)]);
    }
  }

  for (std::size_t row = 0; row < rowLocks.size(); ++row) {
    omp_destroy_lock(&rowLocks[row]);
  }
#else
  for (const int elementId : partition.localElementIds) {
    const Element& element = mesh.elements[static_cast<std::size_t>(elementId)];
    const auto local = element_.computeLocalSystem(mesh, element);

    for (int a = 0; a < 3; ++a) {
      const int globalNodeA = element.nodeIds[a];
      const int dofA = system.nodeToDof[globalNodeA];
      if (dofA < 0) {
        continue;
      }

      const int localRow = system.globalDofToLocalRow[static_cast<std::size_t>(dofA)];
      if (localRow < 0) {
        continue;
      }

      system.rhs[static_cast<std::size_t>(localRow)] += local.load[a];

      for (int b = 0; b < 3; ++b) {
        const int globalNodeB = element.nodeIds[b];
        const int dofB = system.nodeToDof[globalNodeB];
        if (dofB >= 0) {
          system.stiffnessRows[static_cast<std::size_t>(localRow)][dofB] += local.stiffness[a][b];
        } else {
          const Node& boundaryNode = mesh.nodes[globalNodeB];
          system.rhs[static_cast<std::size_t>(localRow)] -=
            local.stiffness[a][b] * exactSolution(boundaryNode.x, boundaryNode.y);
        }
      }
    }
  }
#endif

  return system;
}

}  // namespace laplace
