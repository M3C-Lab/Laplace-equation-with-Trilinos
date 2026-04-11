#include "PostProcessor.hpp"

#include "ProblemData.hpp"

#include <Teuchos_CommHelpers.hpp>
#include <kokkos/Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>

namespace laplace {

namespace detail {

template <class ValueType, class ExecutionSpace>
Kokkos::View<ValueType*, typename ExecutionSpace::array_layout, typename ExecutionSpace::memory_space> copyStdVectorToView(
  const std::vector<ValueType>& input,
  const std::string& label) {
  Kokkos::View<ValueType*, typename ExecutionSpace::array_layout, typename ExecutionSpace::memory_space>
    deviceView(label, input.size());
  auto hostView = Kokkos::create_mirror_view(deviceView);
  for (std::size_t i = 0; i < input.size(); ++i) {
    hostView(static_cast<std::size_t>(i)) = input[i];
  }
  Kokkos::deep_copy(deviceView, hostView);
  return deviceView;
}

KOKKOS_INLINE_FUNCTION
double exactSolutionDevice(const double x, const double y) {
  return sin(kPi * x) * sin(kPi * y);
}

}  // namespace detail

std::vector<double> buildFullSolution(
  const Mesh& mesh,
  const DiscreteSystem& system,
  const DistributedSolution& distributedSolution,
  const Teuchos::RCP<const Teuchos::Comm<int>>& comm) {
  const int numProcs = comm->getSize();
  const int rank = comm->getRank();
  const int localCount = static_cast<int>(distributedSolution.ownedGlobalDofs.size());

  std::vector<int> allCounts(numProcs, 0);
  Teuchos::gatherAll(*comm, 1, &localCount, numProcs, allCounts.data());

  int maxCount = 0;
  for (const int count : allCounts) {
    maxCount = std::max(maxCount, count);
  }

  std::vector<long long> paddedIds(static_cast<std::size_t>(maxCount), -1);
  std::vector<double> paddedValues(static_cast<std::size_t>(maxCount), 0.0);
  for (int i = 0; i < localCount; ++i) {
    paddedIds[static_cast<std::size_t>(i)] = distributedSolution.ownedGlobalDofs[static_cast<std::size_t>(i)];
    paddedValues[static_cast<std::size_t>(i)] = distributedSolution.ownedValues[static_cast<std::size_t>(i)];
  }

  std::vector<long long> gatheredIds(static_cast<std::size_t>(numProcs * maxCount), -1);
  std::vector<double> gatheredValues(static_cast<std::size_t>(numProcs * maxCount), 0.0);
  Teuchos::gatherAll(*comm, maxCount, paddedIds.data(), numProcs * maxCount, gatheredIds.data());
  Teuchos::gatherAll(*comm, maxCount, paddedValues.data(), numProcs * maxCount, gatheredValues.data());

  std::vector<double> interiorSolution(system.nodeToDof.empty() ? 0 : static_cast<std::size_t>(*std::max_element(
    system.nodeToDof.begin(), system.nodeToDof.end()) + 1), 0.0);
  if (rank == 0) {
    for (int proc = 0; proc < numProcs; ++proc) {
      for (int i = 0; i < allCounts[proc]; ++i) {
        const std::size_t offset = static_cast<std::size_t>(proc * maxCount + i);
        const long long globalDof = gatheredIds[offset];
        if (globalDof >= 0) {
          interiorSolution[static_cast<std::size_t>(globalDof)] = gatheredValues[offset];
        }
      }
    }
  }

  Teuchos::broadcast(*comm, 0, static_cast<int>(interiorSolution.size()), interiorSolution.data());

  std::vector<double> nodeX(mesh.nodes.size(), 0.0);
  std::vector<double> nodeY(mesh.nodes.size(), 0.0);
  for (std::size_t nodeId = 0; nodeId < mesh.nodes.size(); ++nodeId) {
    nodeX[nodeId] = mesh.nodes[nodeId].x;
    nodeY[nodeId] = mesh.nodes[nodeId].y;
  }

  using execution_space = AppExecutionSpace;
  auto nodeXView = detail::copyStdVectorToView<double, execution_space>(nodeX, "node_x");
  auto nodeYView = detail::copyStdVectorToView<double, execution_space>(nodeY, "node_y");
  auto nodeToDofView = detail::copyStdVectorToView<int, execution_space>(system.nodeToDof, "node_to_dof");
  auto interiorView = detail::copyStdVectorToView<double, execution_space>(interiorSolution, "interior_solution");
  Kokkos::View<double*, typename execution_space::array_layout, typename execution_space::memory_space>
    fullSolutionView("full_solution", mesh.nodes.size());

  Kokkos::parallel_for(
    "reconstruct_full_solution",
    Kokkos::RangePolicy<execution_space>(0, static_cast<int>(mesh.nodes.size())),
    KOKKOS_LAMBDA(const int nodeId) {
      const int dof = nodeToDofView(static_cast<std::size_t>(nodeId));
      if (dof >= 0) {
        fullSolutionView(static_cast<std::size_t>(nodeId)) = interiorView(static_cast<std::size_t>(dof));
      } else {
        fullSolutionView(static_cast<std::size_t>(nodeId)) =
          detail::exactSolutionDevice(nodeXView(static_cast<std::size_t>(nodeId)),
                                      nodeYView(static_cast<std::size_t>(nodeId)));
      }
    });
  execution_space().fence();

  std::vector<double> fullSolution(mesh.nodes.size(), 0.0);
  auto fullSolutionHost = Kokkos::create_mirror_view(fullSolutionView);
  Kokkos::deep_copy(fullSolutionHost, fullSolutionView);
  for (std::size_t nodeId = 0; nodeId < mesh.nodes.size(); ++nodeId) {
    fullSolution[nodeId] = fullSolutionHost(static_cast<std::size_t>(nodeId));
  }
  return fullSolution;
}

PostProcessSummary computePostProcessSummary(
  const Mesh& mesh,
  const std::vector<double>& solution) {
  PostProcessSummary summary;
  summary.exactValues.resize(mesh.nodes.size(), 0.0);
  summary.errorValues.resize(mesh.nodes.size(), 0.0);

  std::vector<double> nodeX(mesh.nodes.size(), 0.0);
  std::vector<double> nodeY(mesh.nodes.size(), 0.0);
  std::vector<int> elemN0(mesh.elements.size(), 0);
  std::vector<int> elemN1(mesh.elements.size(), 0);
  std::vector<int> elemN2(mesh.elements.size(), 0);
  for (std::size_t nodeId = 0; nodeId < mesh.nodes.size(); ++nodeId) {
    nodeX[nodeId] = mesh.nodes[nodeId].x;
    nodeY[nodeId] = mesh.nodes[nodeId].y;
  }
  for (std::size_t elemId = 0; elemId < mesh.elements.size(); ++elemId) {
    elemN0[elemId] = mesh.elements[elemId].nodeIds[0];
    elemN1[elemId] = mesh.elements[elemId].nodeIds[1];
    elemN2[elemId] = mesh.elements[elemId].nodeIds[2];
  }

  using execution_space = AppExecutionSpace;
  auto nodeXView = detail::copyStdVectorToView<double, execution_space>(nodeX, "post_node_x");
  auto nodeYView = detail::copyStdVectorToView<double, execution_space>(nodeY, "post_node_y");
  auto solutionView = detail::copyStdVectorToView<double, execution_space>(solution, "post_solution");
  auto elemN0View = detail::copyStdVectorToView<int, execution_space>(elemN0, "elem_n0");
  auto elemN1View = detail::copyStdVectorToView<int, execution_space>(elemN1, "elem_n1");
  auto elemN2View = detail::copyStdVectorToView<int, execution_space>(elemN2, "elem_n2");
  Kokkos::View<double*, typename execution_space::array_layout, typename execution_space::memory_space>
    exactView("exact_values", mesh.nodes.size());
  Kokkos::View<double*, typename execution_space::array_layout, typename execution_space::memory_space>
    errorView("error_values", mesh.nodes.size());

  Kokkos::parallel_for(
    "compute_pointwise_error",
    Kokkos::RangePolicy<execution_space>(0, static_cast<int>(mesh.nodes.size())),
    KOKKOS_LAMBDA(const int nodeId) {
      const double exact = detail::exactSolutionDevice(
        nodeXView(static_cast<std::size_t>(nodeId)),
        nodeYView(static_cast<std::size_t>(nodeId)));
      exactView(static_cast<std::size_t>(nodeId)) = exact;
      errorView(static_cast<std::size_t>(nodeId)) =
        solutionView(static_cast<std::size_t>(nodeId)) - exact;
    });

  double maxAbsError = 0.0;
  Kokkos::parallel_reduce(
    "max_abs_error",
    Kokkos::RangePolicy<execution_space>(0, static_cast<int>(mesh.nodes.size())),
    KOKKOS_LAMBDA(const int nodeId, double& localMax) {
      const double value = fabs(errorView(static_cast<std::size_t>(nodeId)));
      if (value > localMax) {
        localMax = value;
      }
    },
    Kokkos::Max<double>(maxAbsError));

  double l2ErrorSquared = 0.0;
  Kokkos::parallel_reduce(
    "l2_error_centroid",
    Kokkos::RangePolicy<execution_space>(0, static_cast<int>(mesh.elements.size())),
    KOKKOS_LAMBDA(const int elemId, double& localSum) {
      const int n0Id = elemN0View(static_cast<std::size_t>(elemId));
      const int n1Id = elemN1View(static_cast<std::size_t>(elemId));
      const int n2Id = elemN2View(static_cast<std::size_t>(elemId));
      const double x0 = nodeXView(static_cast<std::size_t>(n0Id));
      const double y0 = nodeYView(static_cast<std::size_t>(n0Id));
      const double x1 = nodeXView(static_cast<std::size_t>(n1Id));
      const double y1 = nodeYView(static_cast<std::size_t>(n1Id));
      const double x2 = nodeXView(static_cast<std::size_t>(n2Id));
      const double y2 = nodeYView(static_cast<std::size_t>(n2Id));
      const double area = 0.5 * fabs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0));
      const double uhCentroid =
        (solutionView(static_cast<std::size_t>(n0Id)) +
         solutionView(static_cast<std::size_t>(n1Id)) +
         solutionView(static_cast<std::size_t>(n2Id))) / 3.0;
      const double exactCentroid = detail::exactSolutionDevice(
        (x0 + x1 + x2) / 3.0,
        (y0 + y1 + y2) / 3.0);
      const double diff = uhCentroid - exactCentroid;
      localSum += area * diff * diff;
    },
    l2ErrorSquared);
  execution_space().fence();

  summary.maxAbsError = maxAbsError;
  summary.l2Error = std::sqrt(l2ErrorSquared);

  auto exactHost = Kokkos::create_mirror_view(exactView);
  auto errorHost = Kokkos::create_mirror_view(errorView);
  Kokkos::deep_copy(exactHost, exactView);
  Kokkos::deep_copy(errorHost, errorView);
  for (std::size_t nodeId = 0; nodeId < mesh.nodes.size(); ++nodeId) {
    summary.exactValues[nodeId] = exactHost(static_cast<std::size_t>(nodeId));
    summary.errorValues[nodeId] = errorHost(static_cast<std::size_t>(nodeId));
  }

  return summary;
}

void writeCsv(
  const Mesh& mesh,
  const std::vector<double>& solution,
  const PostProcessSummary& summary) {
  std::ofstream csv("build/solution.csv");
  csv << "node_id,x,y,u_h,u_exact,error,is_boundary\n";

  for (std::size_t nodeId = 0; nodeId < mesh.nodes.size(); ++nodeId) {
    const Node& node = mesh.nodes[nodeId];
    csv << nodeId << ','
        << std::setprecision(16) << node.x << ','
        << std::setprecision(16) << node.y << ','
        << std::setprecision(16) << solution[nodeId] << ','
        << std::setprecision(16) << summary.exactValues[nodeId] << ','
        << std::setprecision(16) << summary.errorValues[nodeId] << ','
        << (node.isBoundary ? 1 : 0) << '\n';
  }
}

void writeLegacyVtk(
  const Mesh& mesh,
  const std::vector<double>& solution,
  const PostProcessSummary& summary) {
  std::ofstream vtk("build/solution.vtk");
  vtk << "# vtk DataFile Version 3.0\n";
  vtk << "2D Laplace FEM solution\n";
  vtk << "ASCII\n";
  vtk << "DATASET UNSTRUCTURED_GRID\n";
  vtk << "POINTS " << mesh.nodes.size() << " double\n";
  for (const Node& node : mesh.nodes) {
    vtk << std::setprecision(16) << node.x << ' '
        << std::setprecision(16) << node.y << " 0.0\n";
  }

  vtk << "\nCELLS " << mesh.elements.size() << ' ' << mesh.elements.size() * 4 << '\n';
  for (const Element& element : mesh.elements) {
    vtk << "3 " << element.nodeIds[0] << ' '
        << element.nodeIds[1] << ' '
        << element.nodeIds[2] << '\n';
  }

  vtk << "\nCELL_TYPES " << mesh.elements.size() << '\n';
  for (std::size_t i = 0; i < mesh.elements.size(); ++i) {
    vtk << "5\n";
  }

  vtk << "\nPOINT_DATA " << mesh.nodes.size() << '\n';
  vtk << "SCALARS u_h double 1\n";
  vtk << "LOOKUP_TABLE default\n";
  for (double value : solution) {
    vtk << std::setprecision(16) << value << '\n';
  }

  vtk << "\nSCALARS u_exact double 1\n";
  vtk << "LOOKUP_TABLE default\n";
  for (double value : summary.exactValues) {
    vtk << std::setprecision(16) << value << '\n';
  }

  vtk << "\nSCALARS error double 1\n";
  vtk << "LOOKUP_TABLE default\n";
  for (double value : summary.errorValues) {
    vtk << std::setprecision(16) << value << '\n';
  }
}

void reportPostProcessing(
  const PostProcessSummary& summary,
  std::ostream& out) {
  out << "Approximate nodal max error: " << summary.maxAbsError << '\n';
  out << "Approximate L2 error (centroid quadrature): " << summary.l2Error << '\n';
  out << "Post-processing files:\n";
  out << "  build/solution.csv\n";
  out << "  build/solution.vtk\n";
}

}  // namespace laplace
