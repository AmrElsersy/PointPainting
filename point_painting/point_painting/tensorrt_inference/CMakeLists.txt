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
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})


include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIRS}
    include/
)

# source files includeing .cpp & .cu files
file(GLOB_RECURSE SOURCE_FILES
    cuda_kernels/*.cu
    src/*.cpp
)

# EXECUTABLE
add_executable(convert_onnx_to_engine ${PROJECT_SOURCE_DIR}/convert_onnx_to_engine.cpp)
cuda_add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/main.cpp ${SOURCE_FILES})

target_link_libraries(convert_onnx_to_engine nvinfer nvonnxparser cudart)
target_link_libraries(${PROJECT_NAME} nvinfer nvonnxparser cudart ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES} )
