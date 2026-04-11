# 2D Laplace FEM Example

This project provides a compilable Trilinos skeleton for solving the 2D Laplace
equation on the unit square with linear triangular finite elements.

The current example includes:

- Structured mesh generation on `[0, 1] x [0, 1]`
- P1 triangular element assembly for `-Delta u = f`
- Linear solve with `Tpetra + Belos + Ifpack2`
- Post-processing to CSV and VTK files
- Kokkos-based vector kernels for solution reconstruction and error evaluation
- OpenMP-enabled element assembly on the application side

## Project Layout

The code is organized in a more object-oriented style, similar in spirit to the
PERIGEE example layout:

- `include/Types.hpp`: shared mesh and system data structures
- `include/MeshGenerator.hpp` + `src/MeshGenerator.cpp`: structured triangular mesh generation
- `include/TriangleQuadrature.hpp` + `src/TriangleQuadrature.cpp`: triangle quadrature rule
- `include/TriangleElement.hpp` + `src/TriangleElement.cpp`: P1 triangle local stiffness/load evaluation
- `include/Assembler.hpp` + `src/Assembler.cpp`: global matrix/vector assembly
- `include/LinearSolver.hpp` + `src/LinearSolver.cpp`: Tpetra/Belos/Ifpack2 solver path
- `include/PostProcessor.hpp` + `src/PostProcessor.cpp`: reconstruction, error evaluation, CSV/VTK output
- `src/main.cpp`: lightweight driver

The executable is linked against an internal library `laplace_fem_core`, so the
driver stays small and the FEM components remain independently extensible.

The manufactured solution is:

```text
u(x, y) = sin(pi x) sin(pi y)
```

with homogeneous Dirichlet boundary conditions and matching right-hand side.

## Configure And Build

This repo now provides two user-facing CMake presets:

- `single-rank-openmp`
- `single-rank-cuda`

The OpenMP preset uses the build tree `build/openmp`.
The CUDA preset uses the build tree `build/cuda`.

Example:

```bash
cmake --preset single-rank-openmp
cmake --build --preset single-rank-openmp
```

or

```bash
cmake --preset single-rank-cuda
cmake --build --preset single-rank-cuda
```

If you prefer the older script entry point, it is still available:

```bash
LAPLACE_FEM_BACKEND=OPENMP BUILD_DIR=build/openmp ./scripts/configure.sh
BUILD_DIR=build/openmp ./scripts/build.sh
```

For the CUDA backend:

```bash
LAPLACE_FEM_BACKEND=CUDA BUILD_DIR=build/cuda ./scripts/configure.sh
BUILD_DIR=build/cuda ./scripts/build.sh
```

## Run

Single rank + OpenMP:

```bash
./scripts/run_single_rank_openmp.sh 16 16
```

Single rank + CUDA:

```bash
./scripts/run_single_rank_cuda.sh 16 16
```

The generic scripts are still available if you want direct control:

```bash
BUILD_DIR=build/openmp ./scripts/run_openmp.sh 16 16
BUILD_DIR=build/cuda OMP_NUM_THREADS=1 ./scripts/run_openmp.sh 16 16
```

## Clean CMake Cache

```bash
./scripts/clean_cmake.sh
```

## Parallel Notes

- Matrix assembly uses application-level `OpenMP` pragmas.
- The Tpetra/Kokkos backend is now selectable through `LAPLACE_FEM_BACKEND=OPENMP|CUDA`.
- The current code path is intentionally limited to single-rank execution.
- The previous multi-rank partition option was removed because it did not yet provide a robust general partitioning strategy.
- The current presets use `nvcc_wrapper` because this Trilinos installation exports CUDA-related compile options even for host-side builds.

## Output

The program writes:

- `build/solution.csv`
- `build/solution.vtk`

The VTK file can be opened in ParaView for visualization.
