#pragma once

#include "LinearSolver.hpp"
#include "Types.hpp"

#include <iosfwd>
#include <Teuchos_RCP.hpp>
#include <vector>

namespace laplace {

std::vector<double> buildFullSolution(
  const Mesh& mesh,
  const DiscreteSystem& system,
  const DistributedSolution& distributedSolution,
  const Teuchos::RCP<const Teuchos::Comm<int>>& comm);

PostProcessSummary computePostProcessSummary(
  const Mesh& mesh,
  const std::vector<double>& solution);

void writeCsv(
  const Mesh& mesh,
  const std::vector<double>& solution,
  const PostProcessSummary& summary);

void writeLegacyVtk(
  const Mesh& mesh,
  const std::vector<double>& solution,
  const PostProcessSummary& summary);

void reportPostProcessing(
  const PostProcessSummary& summary,
  std::ostream& out,
  bool wroteFiles);

}  // namespace laplace
