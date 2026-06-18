#include "Preprocessor.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char* argv[]) {
  laplace::PreprocessorOptions options;
  options.outputDirectory = "preprocessed";

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--nx" && i + 1 < argc) {
      options.nx = std::stoi(argv[++i]);
    } else if (arg == "--ny" && i + 1 < argc) {
      options.ny = std::stoi(argv[++i]);
    } else if ((arg == "--parts" || arg == "--partitions") && i + 1 < argc) {
      options.partitions = std::stoi(argv[++i]);
    } else if (arg == "--output-dir" && i + 1 < argc) {
      options.outputDirectory = argv[++i];
    } else if (arg == "--metis-ncommon" && i + 1 < argc) {
      options.metisNCommon = std::stoi(argv[++i]);
    } else if (arg == "--nodal-graph") {
      options.useDualGraph = false;
    } else {
      throw std::runtime_error("Unknown preprocess argument: " + arg);
    }
  }

  if (options.outputDirectory.empty()) {
    throw std::runtime_error("Preprocess output directory must not be empty.");
  }

  laplace::runPreprocessor(options);
  return 0;
}
