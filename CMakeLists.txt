cmake_minimum_required(VERSION 3.18)
project(gpu_diagnostic CUDA CXX)

#Uses C++14 Standard for both CUDA and C++
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# Set my GPU architecture (8.6 = RTX 3060)
set(CMAKE_CUDA_ARCHITECTURES 86)

include_directories(src)

#Create final executable named 'gpu_diagnostic' using those three files
add_executable(gpu_diagnostic
    src/main.cu
    src/bandwidth_test.cu
    src/compute_test.cu
)

#Allows access to CUDA functions
target_link_libraries(gpu_diagnostic cudart)