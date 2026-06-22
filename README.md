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

This repo now provides three user-facing CMake presets:

- `cpu`
- `openmp`
- `cuda`

The CPU/OpenMP/CUDA presets use separate build trees:

- `build-cpu`
- `build-openmp`
- `build-cuda`

Example:

```bash
cmake --preset openmp
cmake --build --preset openmp
```

or

```bash
cmake --preset cuda
cmake --build --preset cuda
```

If you prefer one-click scripts instead of calling the presets manually:

```bash
./scripts/build_cpu.sh
./scripts/build_openmp.sh
./scripts/build_cuda.sh
```

Preprocess a structured mesh and partition it for MPI:

```bash
./scripts/preprocess_mesh.sh 16 16 2
```

The generated mesh files are stored under `Mesh/nx=16_ny=16_np=2/`.

## Run

CPU(MPI):

```bash
./scripts/run_cpu.sh 16 16
```

OpenMP:

```bash
./scripts/run_openmp.sh 16 16
```

CUDA:

```bash
./scripts/run_cuda.sh 16 16
```

## Parallel Notes

- Matrix assembly uses application-level `OpenMP` pragmas.
- The Tpetra/Kokkos backend is now selectable through `LAPLACE_FEM_BACKEND=OPENMP|CUDA`.
- The CPU path supports MPI-partitioned execution using the preprocessed mesh files.
- The previous multi-rank partition option was removed because it did not yet provide a robust general partitioning strategy.
- The current presets use `nvcc_wrapper` because this Trilinos installation exports CUDA-related compile options even for host-side builds.

## CUDA + MPI

 - A "CUDA-aware" mpich is necessary when configured and installed:

 ```bash
 ./configure \
  --prefix=/home/xuanming/lib/mpich-4.3.2 \
  --with-device=ch4:ucx \
  --with-cuda=/usr/local/cuda-13 \
  --enable-shared \
  --enable-cxx \
  --disable-fortran \
  --with-hwloc=embedded \
  CFLAGS=-std=gnu11
 ```

## Output

The program writes solution files into the active mode's build directory:

- `build-cpu/solution.csv` and `build-cpu/solution.vtk`
- `build-openmp/solution.csv` and `build-openmp/solution.vtk`
- `build-cuda/solution.csv` and `build-cuda/solution.vtk`

The VTK file can be opened in ParaView for visualization.
