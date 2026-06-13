cmake_minimum_required(VERSION 3.18)

# Enabling CUDA here gives CMake native CUDA support — no FindCUDA needed.
project(unparallel-parallelism CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Common GPU generations: 75=Turing(RTX20xx), 86=Ampere(RTX30xx), 89=Ada(RTX40xx)
# "native" (CMake 3.24+) auto-detects the machine's GPU instead.
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)

# ── Sources ─────────────────────────────────────────────────────────────────

add_executable(app src/cpp/playground.cpp)

add_executable(vector_add src/cuda/vector_add.cu)
target_include_directories(vector_add PRIVATE src/cuda)

# ── Tests ────────────────────────────────────────────────────────────────────

enable_testing()

add_executable(test_cpp test/cpp/test_playground.cpp)
add_test(NAME test_cpp COMMAND test_cpp)

add_executable(test_vector_add test/cuda/test_vector_add.cu)
target_include_directories(test_vector_add PRIVATE src/cuda)
add_test(NAME test_vector_add COMMAND test_vector_add)
