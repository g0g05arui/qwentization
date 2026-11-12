cmake_minimum_required(VERSION 3.24)
project(reconstruction_bench LANGUAGES CXX CUDA)
find_package(ggml REQUIRED)
find_package(Threads REQUIRED)
add_executable(reconstruct reconstruct.cu)
target_link_libraries(reconstruct PRIVATE ggml::ggml Threads::Threads)
set_target_properties(reconstruct PROPERTIES CXX_STANDARD 17 CUDA_STANDARD 17
  CUDA_ARCHITECTURES 90 BUILD_RPATH_USE_LINK_PATH ON)
target_compile_options(reconstruct PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--ptxas-options=-v>")
