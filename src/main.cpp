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
#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <stdexcept>
#include <vector>

namespace {

std::optional<int> tryParseEnvInt(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return std::nullopt;
  }
  try {
    return std::stoi(value);
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

int detectLocalRank() {
  constexpr const char* candidates[] = {
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "MPI_LOCALRANKID",
    "MV2_COMM_WORLD_LOCAL_RANK",
    "SLURM_LOCALID"
  };

  for (const char* name : candidates) {
    if (const auto value = tryParseEnvInt(name)) {
      return *value;
    }
  }

  return 0;
}

int detectVisibleCudaDeviceCount() {
  if (const auto overrideCount = tryParseEnvInt("LAPLACE_FEM_VISIBLE_GPU_COUNT")) {
    return std::max(1, *overrideCount);
  }

  const char* visibleDevices = std::getenv("CUDA_VISIBLE_DEVICES");
  if (visibleDevices == nullptr || *visibleDevices == '\0') {
    return 1;
  }

  std::stringstream stream(visibleDevices);
  std::string token;
  int count = 0;
  while (std::getline(stream, token, ',')) {
    if (!token.empty()) {
      ++count;
    }
  }

  return std::max(1, count);
}

struct RuntimeInfo {
  int worldRank = 0;
  int worldSize = 1;
  int localRank = 0;
  int assignedGpuId = -1;
  int visibleGpuCount = 0;
  bool mpiCudaSupport = false;
};

class RuntimeScope {
public:
  RuntimeScope(int& argc, char**& argv) {
#if defined(HAVE_TPETRACORE_MPI)
    int mpiInitialized = 0;
    MPI_Initialized(&mpiInitialized);
    if (!mpiInitialized) {
      MPI_Init(&argc, &argv);
      ownsMpi_ = true;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &info_.worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &info_.worldSize);
    info_.localRank = detectLocalRank();
#endif

#if defined(HAVE_TPETRACORE_MPI) && defined(LAPLACE_FEM_BACKEND_CUDA)
    info_.mpiCudaSupport = (MPIX_Query_cuda_support() == 1);
#endif

#if defined(LAPLACE_FEM_BACKEND_CUDA)
    info_.visibleGpuCount = detectVisibleCudaDeviceCount();
    info_.assignedGpuId = info_.localRank % info_.visibleGpuCount;
    Kokkos::InitializationSettings settings;
    settings.set_device_id(info_.assignedGpuId);
    Kokkos::initialize(settings);
#else
    Kokkos::initialize();
#endif
    ownsKokkos_ = true;

#if defined(HAVE_TPETRACORE_MPI)
    Tpetra::initialize(&argc, &argv, MPI_COMM_WORLD);
#else
    Tpetra::initialize(&argc, &argv);
#endif
    ownsTpetra_ = true;
  }

  ~RuntimeScope() {
    if (ownsTpetra_) {
      Tpetra::finalize();
    }
    if (ownsKokkos_) {
      Kokkos::finalize();
    }
#if defined(HAVE_TPETRACORE_MPI)
    if (ownsMpi_) {
      MPI_Finalize();
    }
#endif
  }

  const RuntimeInfo& info() const {
    return info_;
  }

private:
  RuntimeInfo info_;
  bool ownsMpi_ = false;
  bool ownsKokkos_ = false;
  bool ownsTpetra_ = false;
};

}  // namespace

int main(int argc, char* argv[]) {
  RuntimeScope scope(argc, argv);
  const auto& runtime = scope.info();
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
#if defined(LAPLACE_FEM_BACKEND_CUDA)
    *out << "Visible CUDA devices (for this job): " << runtime.visibleGpuCount << '\n';
    *out << "MPI CUDA buffer support: " << (runtime.mpiCudaSupport ? "enabled" : "disabled") << '\n';
    if (comm->getSize() > runtime.visibleGpuCount) {
      *out << "CUDA rank sharing: enabled (" << comm->getSize()
           << " MPI ranks mapped onto " << runtime.visibleGpuCount
           << " visible GPU device(s)).\n";
    }
#endif
    *out << "Benchmark mode: " << (benchmarkMode ? "on" : "off") << '\n';
    *out << "Solver tolerance: " << solverTolerance << '\n';
    *out << "Default preconditioner: " << laplace::defaultPreconditionerName() << '\n';
    *out << "Max iterations: " << laplace::defaultMaxIterations() << '\n';
  }

#if defined(HAVE_TPETRACORE_MPI) && defined(LAPLACE_FEM_BACKEND_CUDA)
  if (comm->getSize() > 1 && !runtime.mpiCudaSupport) {
    throw std::runtime_error(
      "Multi-rank CUDA requires an MPI runtime with CUDA device-buffer support, "
      "but MPIX_Query_cuda_support() reported that CUDA support is disabled.");
  }
#endif

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
    *out << "  assembly kernel = " << assembler.lastAssemblyKernel() << '\n';
  }
  *out << "Rank " << comm->getRank()
       << " owns " << system.ownedGlobalDofs.size()
       << " dof rows, "
       << partition.ownedNodeIds.size() << " owned nodes, "
       << partition.ghostNodeIds.size() << " ghost nodes, and assembles "
       << partition.localElementIds.size()
       << " elements";
#if defined(LAPLACE_FEM_BACKEND_CUDA)
  *out << "; local rank = " << runtime.localRank
       << ", assigned CUDA device = " << runtime.assignedGpuId;
#endif
  *out << ".\n";

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
