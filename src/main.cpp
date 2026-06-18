#include "Assembler.hpp"
#include "LinearSolver.hpp"
#include "PartitionIO.hpp"
#include "PostProcessor.hpp"
#include "Timing.hpp"
#include "TriangleElement.hpp"
#include "TriangleQuadrature.hpp"

#include <Tpetra_Core.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_VerboseObject.hpp>
#include <kokkos/Kokkos_Core.hpp>

#include <algorithm>
#include <iostream>
#include <string>
#include <stdexcept>

int main(int argc, char* argv[]) {
  Tpetra::ScopeGuard scope(&argc, &argv);
  auto comm = Tpetra::getDefaultComm();
  auto out = Teuchos::VerboseObjectBase::getDefaultOStream();

  std::string partitionDirectory = "preprocessed";
  bool benchmarkMode = false;
  double solverTolerance = 1.0e-9;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--partition-dir" && i + 1 < argc) {
      partitionDirectory = argv[++i];
    } else if (arg == "--benchmark") {
      benchmarkMode = true;
    } else if (arg == "--solver-tol" && i + 1 < argc) {
      solverTolerance = std::stod(argv[++i]);
    } else {
      throw std::runtime_error("Unknown runtime argument: " + arg);
    }
  }

  laplace::Timer stageTimer;
  const auto preprocessed = laplace::readPreprocessedMesh(partitionDirectory);
  const auto partition = laplace::readMeshPartition(partitionDirectory, comm->getRank());
  const double meshLoadSecondsLocal = stageTimer.elapsedSeconds();
  const laplace::Mesh& mesh = preprocessed.mesh;

  if (partition.size != comm->getSize()) {
    throw std::runtime_error("MPI size does not match the preprocessed partition count.");
  }

  if (comm->getRank() == 0) {
#if defined(LAPLACE_FEM_ENABLE_OPENMP)
    constexpr const char* kAppOpenMpStatus = "enabled";
#else
    constexpr const char* kAppOpenMpStatus = "disabled";
#endif
    *out << "Loading a preprocessed P1 triangular FEM mesh on the unit square.\n";
    *out << "Requested subdivisions: nx = " << preprocessed.metadata.nx
         << ", ny = " << preprocessed.metadata.ny << '\n';
    *out << "Partition directory: " << partitionDirectory << '\n';
    *out << "Selected backend: " << laplace::kBackendName << '\n';
    *out << "Kokkos default execution space: " << Kokkos::DefaultExecutionSpace::name() << '\n';
    *out << "Application OpenMP assembly: " << kAppOpenMpStatus << '\n';
    *out << "Application OpenMP threads: " << laplace::getApplicationThreadCount() << '\n';
    *out << "MPI ranks: " << comm->getSize() << '\n';
    *out << "Benchmark mode: " << (benchmarkMode ? "on" : "off") << '\n';
    *out << "Solver tolerance: " << solverTolerance << '\n';
  }

  auto maxAcrossRanks = [&](const double localSeconds) {
    double globalSeconds = 0.0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &localSeconds, &globalSeconds);
    return globalSeconds;
  };

  laplace::TimingReport timing;
  timing.add("mesh_load", maxAcrossRanks(meshLoadSecondsLocal));
  stageTimer.reset();
  const laplace::TriangleQuadrature quadrature;
  const laplace::TriangleElement triangleElement(quadrature);
  const laplace::Assembler assembler(triangleElement);
  const laplace::DiscreteSystem system = assembler.assemble(mesh, partition);
  timing.add("assembly", maxAcrossRanks(stageTimer.elapsedSeconds()));
  stageTimer.reset();

  if (comm->getRank() == 0) {
    *out << "Mesh statistics:\n";
    *out << "  nodes    = " << mesh.nodes.size() << '\n';
    *out << "  elements = " << mesh.elements.size() << '\n';
    *out << "  dofs     = "
         << (*std::max_element(system.nodeToDof.begin(), system.nodeToDof.end()) + 1)
         << '\n';
    *out << "  assembly OpenMP threads used = " << assembler.lastAssemblyThreadCount() << '\n';
  }
  *out << "Rank " << comm->getRank()
       << " owns " << system.ownedGlobalDofs.size()
       << " dof rows, "
       << partition.ownedNodeIds.size() << " owned nodes, "
       << partition.ghostNodeIds.size() << " ghost nodes, and assembles "
       << partition.localElementIds.size()
       << " elements.\n";

  using scalar_type = double;
  using local_ordinal_type = int;
  using global_ordinal_type = long long;
  using node_type = laplace::SolverNodeType;

  auto matrix = laplace::buildTpetraMatrix<scalar_type, local_ordinal_type, global_ordinal_type, node_type>(
    system, comm);
  const auto distributedSolution =
    laplace::solveLinearSystem<scalar_type, local_ordinal_type, global_ordinal_type, node_type>(
      system, matrix, comm, *out, solverTolerance);
  timing.add("linear_solve", maxAcrossRanks(stageTimer.elapsedSeconds()));
  const auto fullSolution = laplace::buildFullSolution(mesh, system, distributedSolution, comm);
  const auto summary = laplace::computePostProcessSummary(mesh, fullSolution);

  if (comm->getRank() == 0) {
    const bool wroteFiles = !benchmarkMode;
    if (!benchmarkMode) {
      laplace::writeCsv(mesh, fullSolution, summary);
      laplace::writeLegacyVtk(mesh, fullSolution, summary);
    }
    laplace::reportPostProcessing(summary, *out, wroteFiles);
    *out << "Timing summary (seconds):\n";
    *out << "  mesh_load = " << timing.entries()[0].seconds << '\n';
    *out << "  assembly = " << timing.entries()[1].seconds << '\n';
    *out << "  linear_solve = " << timing.entries()[2].seconds << '\n';
  }

  return 0;
}
