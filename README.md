# GPU Diagnostic Tool

A C++ CUDA-based diagnostic tool that validates GPU performance by running 
memory bandwidth and compute throughput tests on NVIDIA GPUs.

## Tests

**Memory Bandwidth Test**
Allocates a 256MB buffer on the GPU and measures the speed of a memory copy 
operation in GB/s. Validates that measured bandwidth meets a minimum threshold.

**Compute Throughput Test**
Performs a 1024x1024 matrix multiplication using a custom CUDA kernel and 
measures performance in GFLOPS. Validates that compute throughput meets a 
minimum threshold.

## Results

Each test reports:
- Measured value (GB/s or GFLOPS)
- Test duration in milliseconds
- Minimum threshold
- Pass/Fail result

An overall Pass/Fail summary is printed at the end.

## Requirements
- NVIDIA GPU with CUDA Compute Capability 8.6+ (tested on RTX 3060)
- CUDA Toolkit 13.2+
- CMake 3.18+
- Visual Studio 2022 (Windows)

## Build

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## Run

```bash
Release\gpu_diagnostic.exe
```

## Sample Output

```
==============================================
       GPU Diagnostic Tool
==============================================
Bandwidth Test Results:
Measured Bandwidth:        293.14 GB/s
Duration:                  1.83 ms
Minimum Threshold:         150.00 GB/s
Test                       PASS

[Compute Throughput Test]
GFLOPS:     623.86
Duration:   3.442 ms
Threshold:  400.0 GFLOPS
Result:     PASS

==============================================
               SUMMARY
==============================================
  Memory Bandwidth:   PASS (293.14 GB/s)
  Compute Throughput: PASS (623.86 GFLOPS)
  Overall:            PASS
==============================================
```
