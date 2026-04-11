#include "Assembler.hpp"
#include "LinearSolver.hpp"
#include "MeshGenerator.hpp"
#include "PostProcessor.hpp"
#include "Timing.hpp"
#include "TriangleElement.hpp"
#include "TriangleQuadrature.hpp"

#include <Tpetra_Core.hpp>
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

  int nx = 16;
  int ny = 16;
  bool benchmarkMode = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--nx" && i + 1 < argc) {
      nx = std::stoi(argv[++i]);
    } else if (arg == "--ny" && i + 1 < argc) {
      ny = std::stoi(argv[++i]);
    } else if (arg == "--benchmark") {
      benchmarkMode = true;
    } else if (i == 1) {
      nx = std::stoi(arg);
    } else if (i == 2) {
      ny = std::stoi(arg);
    }
  }

  if (comm->getSize() > 1) {
    throw std::runtime_error("This version currently supports single-rank execution only.");
  }

  if (comm->getRank() == 0) {
#if defined(LAPLACE_FEM_ENABLE_OPENMP)
    constexpr const char* kAppOpenMpStatus = "enabled";
#else
    constexpr const char* kAppOpenMpStatus = "disabled";
#endif
    *out << "Building a P1 triangular FEM mesh on the unit square.\n";
    *out << "Requested subdivisions: nx = " << nx << ", ny = " << ny << '\n';
    *out << "Partition mode: serial\n";
    *out << "Selected backend: " << laplace::kBackendName << '\n';
    *out << "Kokkos default execution space: " << Kokkos::DefaultExecutionSpace::name() << '\n';
    *out << "Application OpenMP assembly: " << kAppOpenMpStatus << '\n';
    *out << "Application OpenMP threads: " << laplace::getApplicationThreadCount() << '\n';
    *out << "MPI ranks: " << comm->getSize() << '\n';
    *out << "Benchmark mode: " << (benchmarkMode ? "on" : "off") << '\n';
  }

  laplace::TimingReport timing;
  laplace::Timer stageTimer;

  const laplace::Mesh mesh = laplace::buildStructuredTriangularMesh(nx, ny);
  timing.add("mesh_generation", stageTimer.elapsedSeconds());
  stageTimer.reset();
  const laplace::MeshPartition partition =
    laplace::buildNodeRowPartition(mesh, comm->getRank(), comm->getSize(), false);
  const laplace::TriangleQuadrature quadrature;
  const laplace::TriangleElement triangleElement(quadrature);
  const laplace::Assembler assembler(triangleElement);
  const laplace::DiscreteSystem system = assembler.assemble(mesh, partition);
  timing.add("assembly", stageTimer.elapsedSeconds());
  stageTimer.reset();

  if (comm->getRank() == 0) {
    *out << "Mesh statistics:\n";
    *out << "  nodes    = " << mesh.nodes.size() << '\n';
    *out << "  elements = " << mesh.elements.size() << '\n';
    *out << "  dofs     = "
         << (*std::max_element(system.nodeToDof.begin(), system.nodeToDof.end()) + 1)
         << '\n';
  }
  *out << "Rank " << comm->getRank()
       << " owns " << system.ownedGlobalDofs.size()
       << " rows and assembles " << partition.localElementIds.size()
       << " elements.\n";

  using scalar_type = double;
  using local_ordinal_type = int;
  using global_ordinal_type = long long;
  using node_type = laplace::SolverNodeType;

  auto matrix = laplace::buildTpetraMatrix<scalar_type, local_ordinal_type, global_ordinal_type, node_type>(
    system, comm);
  const auto distributedSolution =
    laplace::solveLinearSystem<scalar_type, local_ordinal_type, global_ordinal_type, node_type>(
      system, matrix, comm, *out);
  timing.add("linear_solve", stageTimer.elapsedSeconds());
  stageTimer.reset();
  const auto fullSolution = laplace::buildFullSolution(mesh, system, distributedSolution, comm);
  const auto summary = laplace::computePostProcessSummary(mesh, fullSolution);
  timing.add("postprocess", stageTimer.elapsedSeconds());
  stageTimer.reset();

  if (comm->getRank() == 0) {
    if (!benchmarkMode) {
      laplace::writeCsv(mesh, fullSolution, summary);
      laplace::writeLegacyVtk(mesh, fullSolution, summary);
    }
    laplace::reportPostProcessing(summary, *out);
    *out << "Timing summary (seconds):\n";
    for (const auto& entry : timing.entries()) {
      *out << "  " << entry.name << " = " << entry.seconds << '\n';
    }
    *out << "TIMER total_seconds " << timing.totalSeconds() << '\n';
  }

  return 0;
}
