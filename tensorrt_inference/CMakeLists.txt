cmake_minimum_required(VERSION 3.5)

PROJECT(pointpainting-tensorrt LANGUAGES CXX)

add_definitions(-std=c++17)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

# CUDA
find_package(CUDA REQUIRED)
set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++17;-g;-G;-gencode;arch=compute_75;code=sm_75)
include_directories(/usr/local/cuda-11.1/include)
link_directories(/usr/local/cuda-11.1/lib64)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

# LAYER PLUGIN LIB
# cuda_add_library(layerplugin SHARED ${PROJECT_SOURCE_DIR}/layers/yololayer.cu)

# EXECUTABLE
add_executable(convert_onnx_to_engine ${PROJECT_SOURCE_DIR}/convert_onnx_to_engine.cpp)
add_executable(main ${PROJECT_SOURCE_DIR}/main.cpp)
target_link_libraries(convert_onnx_to_engine nvinfer nvonnxparser cudart)
target_link_libraries(main nvinfer nvonnxparser cudart)
