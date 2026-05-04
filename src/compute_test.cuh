#pragma once

// Stores the result of the compute throughput test
struct ComputeResult {
    float gflops;         // Measured compute throughput in GFLOPS
    float duration_ms;    // How long the test took in milliseconds
    bool passed;          // Passes if minimum threshold is met
};

// Runs the compute throughput test and returns the result
ComputeResult runComputeTest();