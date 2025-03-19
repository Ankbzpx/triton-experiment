cmake_minimum_required(VERSION 3.14)
project(faster_closest_neighbor LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CUDA_ARCHITECTURES 86)
# Enable the generation of compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# For FetchContent
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(cudaKDTree)
# As much I prefer FetchContent, it does not support recursive submmodules...
set(TCNN_BUILD_BENCHMARK OFF)
set(TCNN_BUILD_EXAMPLES OFF)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/third_party/tiny-cuda-nn)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Adapted from https://github.com/shenmishajing/pytorch_extension_example/tree/main
set(Torch_DIR "/home/ruichenzheng/anaconda3/envs/triton/lib/python3.10/site-packages/torch/share/cmake/Torch")
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

set(HDR_DIR include)
set(SRC_DIR src)

set(HDRS
    ${HDR_DIR}/closest_query.cuh
)

set(SRCS
    ${SRC_DIR}/closest_query.cu
    ${CMAKE_CURRENT_LIST_DIR}/main.cu
)

add_executable(${PROJECT_NAME} ${HDRS} ${SRCS})
target_include_directories(${PROJECT_NAME} PRIVATE ${HDR_DIR})
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${Python_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${Python_LIBRARIES} ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} cudaKDTree tiny-cuda-nn)