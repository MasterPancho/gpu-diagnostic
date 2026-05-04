#include "compute_test.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

// Matrix size (1024x1024)
#define MATRIX_SIZE 1024

// Minimum acceptable GFLOPS for RTX 3060 Laptop GPU
// FLOPS = Floating Point Operations Per Second (measure of how much math the GPU is doing every second.)
// Theoretical max is ~13,000 GFLOPS, set a minimum to account for naive implementation overhead
#define MIN_GFLOPS 400.0f

// CUDA kernel --> performs matrix multiplication C = A * B
// Each thread computes one element of the output matrix C
__global__ void matrixMulKernel(float* A, float* B, float* C, int n) {
    
    // Calculate row and column index for this thread (We can have 2D blocks (1024x1024), and each block has 2D threads (16x16))
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure thread is within matrix bounds (launched threads may exceed matrix size)
    if (col < n && row < n){
        float sum = 0.0f;
        for(int k=0; k < n; k++){
            // Accumulate dot product: walk across row of A and down column of B simultaneously
            sum += A[row * n + k] * B[k * n + col];
        }

        // Store the computed dot product into the corresponding element of output matrix C
        C[row*n + col] = sum;
    }
}

ComputeResult runComputeTest() {
    ComputeResult result = {0.0f, 0.0f, false};

    int n = MATRIX_SIZE;
    size_t matrixBytes = n * n * sizeof(float);

    // Allocate CPU matrices and fill them with test data
    float* h_A = new float[n*n];
    float* h_B = new float[n*n];
    for(int k=0; k < n*n; k++){
        h_A[k] = (float)k;
        h_B[k] = (float)k;
    }

    // Allocate GPU matrices
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    cudaMalloc(&d_A, matrixBytes);
    cudaMalloc(&d_B, matrixBytes);
    cudaMalloc(&d_C, matrixBytes);

    // Copy input matrices from CPU to GPU
    cudaMemcpy(d_A, h_A, matrixBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixBytes, cudaMemcpyHostToDevice);


    // Configure kernel launch w/ 2D grid
    // Each block is 16x16 threads (256 threads total, same as bandwidth test)
    dim3 threadsPerBlock(16, 16);

    //1024x1024 matrix with 16x16 blocks
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //Launch the kernel and measure the time
    cudaEventRecord(start);
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    //Calculate the elapsed time
    float ms = 0;

    //Calculate the elapsed time in milliseconds 
    cudaEventElapsedTime(&ms, start, stop);

    //Calculate GFLOPS
    // total number of operations in the matrix multiplication
    float totalOps = 2.0f * n * n * n;
    float gflops = (totalOps / (ms / 1000.0f)) / 1e9f;

    result.gflops = gflops;
    result.duration_ms = ms;
    result.passed = (gflops >= MIN_GFLOPS);

    // Print results
    printf("\n[Compute Throughput Test]\n");
    printf("GFLOPS:     %.2f\n", gflops);
    printf("Duration:   %.3f ms\n", ms);
    printf("Threshold:  %.1f GFLOPS\n", MIN_GFLOPS);
    printf("Result:     %s\n", result.passed ? "PASS" : "FAIL");

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;

    return result;
}

