#include "bandwidth_test.cuh"
#include "cuda_runtime.h"
#include <iostream>

// How much memory to test with (256MB)
#define BUFFER_SIZE (256 * 1024 * 1024)

// Minimum acceptable bandwidth for RTX 3060 Laptop GPU in GB/s. 
// Theoretical max is ~336 GB/s, we set threshold at 50% to account for overhead
#define MIN_BANDWIDTH_GBS 150.0f

// CUDA kernel to perform memory copy between two buffers
__global__ void bandwidthKernel(float* iBuffer, float* oBuffer, int numElements){
    // Different idx for the given thread. Ex:Block 1, Thread 2, maxTHreads = 256, idx = 1*256 + 2 = 258 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  

    // Copy data from input to output buffer
    if (idx < numElements){
        oBuffer[idx] = iBuffer[idx];
    }
}

BandwidthResult runBandwidthTest(){
    BandwidthResult result = {0.0f, 0.0f, false};
    
    //The data is stored as floats, so we need to calculate how many float elements fit into our buffer 
    int numElements = BUFFER_SIZE / sizeof(float);   
    
    float* d_input = nullptr;
    float* d_output = nullptr;

    // Allocate memory on the GPU for input and output buffers
    cudaMalloc(&d_input, BUFFER_SIZE);
    cudaMalloc(&d_output, BUFFER_SIZE);

    // Allocate memory on the host for input buffer and assign values to it
    float* h_input = new float[numElements];
    for (int i=0; i < numElements; i++){
        h_input[i] = (float)i;
    }

    //Copy the data from the CPU input variable to the GPU input variable.
    cudaMemcpy(d_input, h_input, BUFFER_SIZE, cudaMemcpyHostToDevice);

    // We could technically use this instead, but using the other method would also allow us for data integrity check.
    // Each element is being initialized to its index value, so once the kernel runs, d_output[i] must equal i, 
    // making any corruption detectable.
    // cudaMemset(d_input, 1, BUFFER_SIZE);

    // Configure kernel launch parameters (Note: We add numElements with threadsPerBlock in order to round the number of blocks up)
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    //Start recording the time
    cudaEventRecord(start);

    // <<<>>> --> CUDA syntax for launching a kernel on the GPU. "Take this function and run it across this many blocks and threads"
    // Each of the 64M threads copies one float element from d_input to d_output simultaneously
    bandwidthKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, numElements);

    //Stop recording the time and wait for the kernel to finish
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    //Calculate the elapsed time in milliseconds and compute the bandwidth in GB/s (Note: 2.0f*BUFFER_SIZE because we read and write)
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    result.bandwidth_GBs = (2.0f * BUFFER_SIZE) / (milliseconds / 1000.0f) / 1e9f;
    result.duration_ms = milliseconds;

    //Check if the measured bandwidth meets the minimum threshold
    result.passed = (result.bandwidth_GBs >= MIN_BANDWIDTH_GBS);

    // Results
    printf("Bandwidth Test Results:\n");
    printf("Measured Bandwidth:     %.2f GB/s\n", result.bandwidth_GBs);
    printf("Duration:               %.2f ms\n", result.duration_ms);
    printf("Minimum Threshold:      %.2f GB/s\n", MIN_BANDWIDTH_GBS);
    printf("Result:                 %s\n", result.passed ? "PASSED" : "FAILED");

    //Clean up resources
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    
    return result;
}