#include "LinearSolver.hpp"

#include <BelosLinearProblem.hpp>
#include <BelosPseudoBlockCGSolMgr.hpp>
#include <BelosTpetraAdapter.hpp>
#include <Ifpack2_Factory.hpp>
#include <Tpetra_Vector.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>
#include <kokkos/Kokkos_Core.hpp>

#include <algorithm>
#include <stdexcept>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace laplace {

namespace detail {

template <class ValueType, class DeviceType>
Kokkos::View<ValueType*, DeviceType> copyStdVectorToView(
  const std::vector<ValueType>& input,
  const std::string& label) {
  Kokkos::View<ValueType*, DeviceType> deviceView(label, input.size());
  auto hostView = Kokkos::create_mirror_view(deviceView);
  for (std::size_t i = 0; i < input.size(); ++i) {
    hostView(static_cast<std::size_t>(i)) = input[i];
  }
  Kokkos::deep_copy(deviceView, hostView);
  return deviceView;
}

template <class VectorType>
void fillVectorFromStdVector(
  const std::vector<double>& input,
  VectorType& vector) {
  using device_type = typename VectorType::device_type;
  auto inputView = copyStdVectorToView<double, device_type>(input, "rhs_input");
  auto vectorView = vector.getLocalViewDevice(Tpetra::Access::ReadWrite);
  using execution_space = typename device_type::execution_space;
  Kokkos::parallel_for(
    "fill_vector_from_std",
    Kokkos::RangePolicy<execution_space>(0, static_cast<int>(input.size())),
    KOKKOS_LAMBDA(const int i) {
      vectorView(i, 0) = inputView(static_cast<std::size_t>(i));
    });
  execution_space().fence();
}

}  // namespace detail

int getApplicationThreadCount() {
#if defined(_OPENMP)
  return omp_get_max_threads();
#else
  return 1;
#endif
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class NodeType>
Teuchos::RCP<Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, NodeType>>
buildTpetraMatrix(
  const DiscreteSystem& system,
  const Teuchos::RCP<const Teuchos::Comm<int>>& comm) {
  using matrix_type = Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, NodeType>;
  using map_type = Tpetra::Map<LocalOrdinal, GlobalOrdinal, NodeType>;

  const auto numDofs = static_cast<Tpetra::global_size_t>(
    *std::max_element(system.nodeToDof.begin(), system.nodeToDof.end()) + 1);
  Teuchos::Array<GlobalOrdinal> ownedRows;
  ownedRows.reserve(system.ownedGlobalDofs.size());
  for (const long long globalDof : system.ownedGlobalDofs) {
    ownedRows.push_back(static_cast<GlobalOrdinal>(globalDof));
  }

  auto map = Teuchos::rcp(new map_type(
    numDofs,
    ownedRows(),
    static_cast<GlobalOrdinal>(0),
    comm));
  auto matrix = Teuchos::rcp(new matrix_type(map, 7));

  for (std::size_t localRow = 0; localRow < system.stiffnessRows.size(); ++localRow) {
    Teuchos::Array<GlobalOrdinal> columns;
    Teuchos::Array<Scalar> values;

    for (const auto& [col, value] : system.stiffnessRows[localRow]) {
      columns.push_back(static_cast<GlobalOrdinal>(col));
      values.push_back(static_cast<Scalar>(value));
    }

    matrix->insertGlobalValues(
      static_cast<GlobalOrdinal>(system.ownedGlobalDofs[localRow]),
      columns(),
      values());
  }

  matrix->fillComplete();
  return matrix;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class NodeType>
DistributedSolution solveLinearSystem(
  const DiscreteSystem& system,
  const Teuchos::RCP<Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, NodeType>>& matrix,
  const Teuchos::RCP<const Teuchos::Comm<int>>& comm,
  std::ostream& out,
  const double solverTolerance) {
  using matrix_type = Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, NodeType>;
  using map_type = Tpetra::Map<LocalOrdinal, GlobalOrdinal, NodeType>;
  using vector_type = Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, NodeType>;
  using operator_type = Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, NodeType>;
  using multivector_type = Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, NodeType>;
  using problem_type = Belos::LinearProblem<Scalar, multivector_type, operator_type>;

  const auto numDofs = static_cast<Tpetra::global_size_t>(
    *std::max_element(system.nodeToDof.begin(), system.nodeToDof.end()) + 1);
  Teuchos::Array<GlobalOrdinal> ownedRows;
  ownedRows.reserve(system.ownedGlobalDofs.size());
  for (const long long globalDof : system.ownedGlobalDofs) {
    ownedRows.push_back(static_cast<GlobalOrdinal>(globalDof));
  }

  auto map = Teuchos::rcp(new map_type(
    numDofs,
    ownedRows(),
    static_cast<GlobalOrdinal>(0),
    comm));
  auto lhs = Teuchos::rcp(new vector_type(map));
  auto rhs = Teuchos::rcp(new vector_type(map));
  lhs->putScalar(0.0);
  detail::fillVectorFromStdVector(system.rhs, *rhs);

  auto problem = Teuchos::rcp(new problem_type(matrix, lhs, rhs));

  Ifpack2::Factory factory;
  auto preconditioner = factory.template create<matrix_type>("RELAXATION", matrix);
  Teuchos::ParameterList precParams;
  precParams.set("relaxation: type", "Symmetric Gauss-Seidel");
  precParams.set("relaxation: sweeps", 2);
  precParams.set("relaxation: damping factor", 1.0);
  preconditioner->setParameters(precParams);
  preconditioner->initialize();
  preconditioner->compute();
  problem->setRightPrec(preconditioner);

  if (!problem->setProblem()) {
    throw std::runtime_error("Belos failed to accept the linear problem.");
  }

  Teuchos::ParameterList solverParams;
  solverParams.set("Maximum Iterations", 1000);
  solverParams.set("Convergence Tolerance", solverTolerance);
  solverParams.set("Verbosity", Belos::Errors + Belos::Warnings + Belos::FinalSummary);
  solverParams.set("Output Frequency", 1);
  solverParams.set("Output Style", 1);

  Belos::PseudoBlockCGSolMgr<Scalar, multivector_type, operator_type> solver(
    problem, Teuchos::rcpFromRef(solverParams));
  const Belos::ReturnType result = solver.solve();

  if (result != Belos::Converged) {
    throw std::runtime_error("Belos CG did not converge.");
  }

  DistributedSolution solution;
  solution.ownedGlobalDofs = system.ownedGlobalDofs;
  solution.ownedValues.resize(system.rhs.size(), 0.0);
  auto solutionView = lhs->getLocalViewDevice(Tpetra::Access::ReadOnly);
  auto solutionHost = Kokkos::create_mirror_view(solutionView);
  Kokkos::deep_copy(solutionHost, solutionView);
  for (std::size_t i = 0; i < solution.ownedValues.size(); ++i) {
    solution.ownedValues[i] = solutionHost(static_cast<LocalOrdinal>(i), 0);
  }

  out << "Belos iterations: " << solver.getNumIters() << '\n';
  return solution;
}

template
Teuchos::RCP<Tpetra::CrsMatrix<double, int, long long, SolverNodeType>>
buildTpetraMatrix<double, int, long long, SolverNodeType>(
  const DiscreteSystem& system,
  const Teuchos::RCP<const Teuchos::Comm<int>>& comm);

template
DistributedSolution solveLinearSystem<double, int, long long, SolverNodeType>(
  const DiscreteSystem& system,
  const Teuchos::RCP<Tpetra::CrsMatrix<double, int, long long, SolverNodeType>>& matrix,
  const Teuchos::RCP<const Teuchos::Comm<int>>& comm,
  std::ostream& out,
  double solverTolerance);

}  // namespace laplace
