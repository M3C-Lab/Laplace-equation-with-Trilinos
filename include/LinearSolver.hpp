#pragma once

#include "Types.hpp"

#include <kokkos/Kokkos_Core.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_KokkosCompat_ClassicNodeAPI_Wrapper.hpp>
#include <Tpetra_Map.hpp>

#include <iosfwd>
#include <vector>

namespace laplace {

struct DistributedSolution {
  std::vector<long long> ownedGlobalDofs;
  std::vector<double> ownedValues;
};

#if defined(LAPLACE_FEM_BACKEND_OPENMP)
using SolverNodeType = Tpetra::KokkosCompat::KokkosOpenMPWrapperNode;
using AppExecutionSpace = Kokkos::OpenMP;
inline constexpr const char* kBackendName = "OPENMP";
#elif defined(LAPLACE_FEM_BACKEND_CUDA)
using SolverNodeType = Tpetra::KokkosCompat::KokkosCudaWrapperNode;
using AppExecutionSpace = Kokkos::Cuda;
inline constexpr const char* kBackendName = "CUDA";
#else
#error "No Laplace backend selected. Define LAPLACE_FEM_BACKEND_OPENMP or LAPLACE_FEM_BACKEND_CUDA."
#endif

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class NodeType>
Teuchos::RCP<Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, NodeType>>
buildTpetraMatrix(
  const DiscreteSystem& system,
  const Teuchos::RCP<const Teuchos::Comm<int>>& comm);

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class NodeType>
DistributedSolution solveLinearSystem(
  const DiscreteSystem& system,
  const Teuchos::RCP<Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, NodeType>>& matrix,
  const Teuchos::RCP<const Teuchos::Comm<int>>& comm,
  std::ostream& out,
  double solverTolerance = 1.0e-9);

int getApplicationThreadCount();

}  // namespace laplace
