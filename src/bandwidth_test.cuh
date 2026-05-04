#pragma once

// Stores the result of the bandwidth test
struct BandwidthResult {
    float bandwidth_GBs;    // Measured memory bandwidth in GB/s
    float duration_ms;      // How long the test took in milliseconds
    bool passed;            // True if minimum threshold met
};

// Runs the memory bandwidth test and returns the result
BandwidthResult runBandwidthTest();