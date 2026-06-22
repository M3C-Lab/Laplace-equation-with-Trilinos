cmake_minimum_required(VERSION 3.20)

# Prefer the MPI compiler wrapper that matches the local Trilinos install.
if(NOT DEFINED CMAKE_CXX_COMPILER)
  find_program(LAPLACE_FEM_MPICXX
    NAMES mpicxx
    HINTS /home/xuanming/lib/mpich-4.3.2/bin
  )
  if(LAPLACE_FEM_MPICXX)
    set(CMAKE_CXX_COMPILER "${LAPLACE_FEM_MPICXX}" CACHE FILEPATH "C++ compiler" FORCE)
  endif()
endif()

project(TrilinosLaplaceFEM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(Trilinos_ROOT "/home/xuanming/lib/Trilinos" CACHE PATH "Path to the Trilinos installation prefix")
set(METIS_ROOT "/home/xuanming/lib/metis-5.0.3" CACHE PATH "Path to the METIS installation prefix")
set(HDF5_ROOT "/home/xuanming/lib/hdf5-1.12.2" CACHE PATH "Path to the HDF5 installation prefix")
list(PREPEND CMAKE_PREFIX_PATH "${Trilinos_ROOT}")

# This application uses Kokkos through Trilinos but does not compile CUDA
# sources directly, so prefer the host MPI compiler and avoid global
# redirection through nvcc_wrapper during the local build.
set(Kokkos_LAUNCH_COMPILER OFF CACHE BOOL "Disable Kokkos global compile launcher" FORCE)

option(LAPLACE_FEM_ENABLE_OPENMP "Enable OpenMP in the application assembly kernels" ON)
set(LAPLACE_FEM_BACKEND "OPENMP" CACHE STRING "Backend for Tpetra/Kokkos kernels: OPENMP or CUDA")
set_property(CACHE LAPLACE_FEM_BACKEND PROPERTY STRINGS OPENMP CUDA)

if(NOT LAPLACE_FEM_BACKEND STREQUAL "OPENMP" AND
   NOT LAPLACE_FEM_BACKEND STREQUAL "CUDA")
  message(FATAL_ERROR "LAPLACE_FEM_BACKEND must be OPENMP or CUDA")
endif()

find_package(Trilinos REQUIRED COMPONENTS Teuchos Tpetra Belos Ifpack2)
find_package(OpenMP QUIET)
find_package(HDF5 REQUIRED)
find_path(METIS_INCLUDE_DIR metis.h HINTS "${METIS_ROOT}/include")
find_library(METIS_LIBRARY metis HINTS "${METIS_ROOT}/lib")

if(NOT METIS_INCLUDE_DIR OR NOT METIS_LIBRARY)
  message(FATAL_ERROR "Failed to locate METIS under ${METIS_ROOT}")
endif()

set(LAPLACE_FEM_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${Trilinos_INCLUDE_DIRS}
  ${METIS_INCLUDE_DIR}
  ${HDF5_INCLUDE_DIRS}
)


set(laplace_fem_core_src
  src/ProblemData.cpp
  src/HDF5_IO.cpp
  src/MeshGenerator.cpp
  src/PartitionIO.cpp
  src/Preprocessor.cpp
  src/TriangleQuadrature.cpp
  src/TriangleElement.cpp
  src/ElementKernel.cpp
  src/Assembler.cpp
  src/LinearSolver.cpp
  src/PostProcessor.cpp
)

add_library(laplace_fem_core ${laplace_fem_core_src})

if(CMAKE_CXX_COMPILER MATCHES "nvcc_wrapper$")
  set_source_files_properties(
    src/Assembler.cpp
    PROPERTIES
      COMPILE_OPTIONS "--host-only"
  )
endif()

target_include_directories(laplace_fem_core
  PRIVATE
    ${LAPLACE_FEM_INCLUDE_DIRS}
)

target_include_directories(laplace_fem_core
  PUBLIC
    ${LAPLACE_FEM_INCLUDE_DIRS}
)

if(LAPLACE_FEM_BACKEND STREQUAL "OPENMP")
  target_compile_definitions(laplace_fem_core PUBLIC LAPLACE_FEM_BACKEND_OPENMP=1)
else()
  target_compile_definitions(laplace_fem_core PUBLIC LAPLACE_FEM_BACKEND_CUDA=1)
endif()

if(LAPLACE_FEM_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(laplace_fem_core PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(laplace_fem_core PUBLIC LAPLACE_FEM_ENABLE_OPENMP=1)
  if(OpenMP_CXX_INCLUDE_DIRS)
    target_include_directories(laplace_fem_core PUBLIC ${OpenMP_CXX_INCLUDE_DIRS})
  endif()
  message(STATUS "Application OpenMP support: enabled")
else()
  message(STATUS "Application OpenMP support: disabled")
endif()

target_link_libraries(laplace_fem_core
  PUBLIC
    Teuchos::all_libs
    Tpetra::all_libs
    Belos::all_libs
    Ifpack2::all_libs
    ${METIS_LIBRARY}
    ${HDF5_LIBRARIES}
)

add_executable(laplace_fem_2d
  src/main.cpp
)

add_executable(laplace_preprocess
  src/preprocess_main.cpp
)

target_include_directories(laplace_fem_2d
  PRIVATE
    ${LAPLACE_FEM_INCLUDE_DIRS}
)

target_include_directories(laplace_preprocess
  PRIVATE
    ${LAPLACE_FEM_INCLUDE_DIRS}
)

target_link_libraries(laplace_fem_2d
  PRIVATE
    laplace_fem_core
)

target_link_libraries(laplace_preprocess
  PRIVATE
    laplace_fem_core
)

message(STATUS "Found Trilinos version: ${Trilinos_VERSION}")
message(STATUS "Using Trilinos from: ${Trilinos_INSTALL_DIR}")
message(STATUS "Selected application backend: ${LAPLACE_FEM_BACKEND}")
