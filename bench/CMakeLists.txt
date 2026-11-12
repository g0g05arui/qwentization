cmake_minimum_required(VERSION 3.14)
project(qwentization_bench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ggml from an installed llama.cpp / ggml tree:
#   cmake -B build -DCMAKE_PREFIX_PATH=/path/to/ggml-install
find_package(ggml REQUIRED)

add_executable(matvec_energy matvec_energy.cpp)
target_link_libraries(matvec_energy PRIVATE ggml::ggml ${CMAKE_DL_LIBS} Threads::Threads)
find_package(Threads REQUIRED)

# find the ggml shared libraries / backends at run time without LD_LIBRARY_PATH
set_target_properties(matvec_energy PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON BUILD_RPATH_USE_LINK_PATH ON)
