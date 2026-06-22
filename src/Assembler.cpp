#include "Assembler.hpp"

#include "ElementKernel.hpp"
#include "ProblemData.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <stdexcept>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace laplace {

namespace {

std::string uppercaseAscii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  return value;
}

bool useKokkosElementAssembly() {
#if defined(LAPLACE_FEM_BACKEND_CUDA)
  if (const char* env = std::getenv("LAPLACE_FEM_ASSEMBLY_KERNEL")) {
    const std::string value = uppercaseAscii(env);
    if (value == "HOST") {
      return false;
    }
    if (value == "KOKKOS" || value == "KOKKOS_ELEMENT") {
      return true;
    }
  }
  return false;
#else
  return false;
#endif
}

}  // namespace

Assembler::Assembler(const TriangleElement& element)
    : element_(element) {}

int Assembler::lastAssemblyThreadCount() const {
  return lastAssemblyThreadCount_;
}

const char* Assembler::lastAssemblyKernel() const {
  return lastAssemblyKernel_;
}

DiscreteSystem Assembler::assemble(const Mesh& mesh, const MeshPartition& partition) const {
  lastAssemblyThreadCount_ = 1;
  lastAssemblyKernel_ = "host";
  DiscreteSystem system;
  system.nodeToDof.assign(mesh.nodes.size(), -1);

  int dofCount = 0;
  for (std::size_t nodeId = 0; nodeId < mesh.nodes.size(); ++nodeId) {
    if (!mesh.nodes[nodeId].isBoundary) {
      system.nodeToDof[nodeId] = dofCount++;
    }
  }

  system.globalDofToLocalRow.assign(static_cast<std::size_t>(dofCount), -1);

  for (const int nodeId : partition.ownedNodeIds) {
    const int dof = system.nodeToDof[static_cast<std::size_t>(nodeId)];
    if (dof >= 0) {
      system.globalDofToLocalRow[static_cast<std::size_t>(dof)] =
        static_cast<int>(system.ownedGlobalDofs.size());
      system.ownedGlobalDofs.push_back(static_cast<long long>(dof));
      system.ownedNodeIds.push_back(nodeId);
    }
  }

  system.stiffnessRows.resize(system.ownedGlobalDofs.size());
  system.rhs.assign(system.ownedGlobalDofs.size(), 0.0);

#if defined(LAPLACE_FEM_BACKEND_CUDA)
  if (useKokkosElementAssembly()) {
  lastAssemblyKernel_ = "kokkos-element";
  const std::size_t localElementCount = partition.localElementIds.size();
  const auto kernelOutput = computeElementLocalSystemsKokkos(partition);

  for (std::size_t localElemIndex = 0; localElemIndex < localElementCount; ++localElemIndex) {
    const int elementId = partition.localElementIds[localElemIndex];
    const Element& element = mesh.elements[static_cast<std::size_t>(elementId)];
    for (int a = 0; a < 3; ++a) {
      const int globalNodeA = element.nodeIds[a];
      const int dofA = system.nodeToDof[static_cast<std::size_t>(globalNodeA)];
      if (dofA < 0) {
        continue;
      }

      const int localRow = system.globalDofToLocalRow[static_cast<std::size_t>(dofA)];
      if (localRow < 0) {
        continue;
      }

      const std::size_t loadBase = localElemIndex * 3;
      const std::size_t stiffnessBase = localElemIndex * 9;
      system.rhs[static_cast<std::size_t>(localRow)] +=
        kernelOutput.load[loadBase + static_cast<std::size_t>(a)];
      for (int b = 0; b < 3; ++b) {
        const int globalNodeB = element.nodeIds[b];
        const int dofB = system.nodeToDof[static_cast<std::size_t>(globalNodeB)];
        if (dofB >= 0) {
          system.stiffnessRows[static_cast<std::size_t>(localRow)][dofB] +=
            kernelOutput.stiffness[stiffnessBase + static_cast<std::size_t>(a * 3 + b)];
        } else {
          const Node& boundaryNode = mesh.nodes[static_cast<std::size_t>(globalNodeB)];
          system.rhs[static_cast<std::size_t>(localRow)] -=
            kernelOutput.stiffness[stiffnessBase + static_cast<std::size_t>(a * 3 + b)] *
            exactSolution(boundaryNode.x, boundaryNode.y);
        }
      }
    }
  }

  return system;
  }
#endif

#if defined(_OPENMP)
  std::vector<omp_lock_t> rowLocks(system.ownedGlobalDofs.size());
  for (std::size_t row = 0; row < rowLocks.size(); ++row) {
    omp_init_lock(&rowLocks[row]);
  }
#endif

#if defined(_OPENMP)
#pragma omp parallel
  {
#pragma omp single
    lastAssemblyThreadCount_ = omp_get_num_threads();

#pragma omp for schedule(static)
    for (std::ptrdiff_t elem = 0; elem < static_cast<std::ptrdiff_t>(partition.localElementIds.size()); ++elem) {
      const std::size_t localElemIndex = static_cast<std::size_t>(elem);
      const int elementId = partition.localElementIds[localElemIndex];
      const auto& lien = partition.lien[localElemIndex];
      const std::array<Node, 3> localNodes{
        partition.localNodes[static_cast<std::size_t>(lien[0])],
        partition.localNodes[static_cast<std::size_t>(lien[1])],
        partition.localNodes[static_cast<std::size_t>(lien[2])]
      };
      const Element& element = mesh.elements[static_cast<std::size_t>(elementId)];
      const auto local = element_.computeLocalSystem(localNodes);

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
  }

  for (std::size_t row = 0; row < rowLocks.size(); ++row) {
    omp_destroy_lock(&rowLocks[row]);
  }
#else
  for (std::size_t localElemIndex = 0; localElemIndex < partition.localElementIds.size(); ++localElemIndex) {
    const int elementId = partition.localElementIds[localElemIndex];
    const auto& lien = partition.lien[localElemIndex];
    const std::array<Node, 3> localNodes{
      partition.localNodes[static_cast<std::size_t>(lien[0])],
      partition.localNodes[static_cast<std::size_t>(lien[1])],
      partition.localNodes[static_cast<std::size_t>(lien[2])]
    };
    const Element& element = mesh.elements[static_cast<std::size_t>(elementId)];
    const auto local = element_.computeLocalSystem(localNodes);

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
