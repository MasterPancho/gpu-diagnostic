#include <stdio.h>
#include "bandwidth_test.cuh"
#include "compute_test.cuh"

int main(){

    printf("==============================================\n");
    printf("       GPU Diagnostic Tool\n");
    printf("==============================================\n");

    // Run bandwidth test and print results
    BandwidthResult bResult = runBandwidthTest();

    // Run compute test and print results
    ComputeResult cResult = runComputeTest();

    // Overall summary
    printf("\n==============================================\n");
    printf("               SUMMARY\n");
    printf("==============================================\n");
    printf("  Memory Bandwidth:  %s (%.2f GB/s)\n", bResult.passed ? "PASS" : "FAIL", bResult.bandwidth_GBs);
    printf("  Compute Throughput: %s (%.2f GFLOPS)\n", cResult.passed ? "PASS" : "FAIL", cResult.gflops);
    printf("  Overall:           %s\n", (bResult.passed && cResult.passed) ? "PASS" : "FAIL");
    printf("==============================================\n");


    // Write results to CSV file
    FILE* csvFile = fopen("../results/diagnostic_report.csv", "w");
    if (csvFile != nullptr) {
        fprintf(csvFile, "Test,Value,Threshold,Result,Duration(ms)\n");
        
        // Write bandwidth test result
        fprintf(csvFile, "Memory Bandwidth,%.2f GB/s,%.2f GB/s,%s,%.3f\n",
            bResult.bandwidth_GBs, 150.0f,
            bResult.passed ? "PASS" : "FAIL",
            bResult.duration_ms);

        // Write compute test result
        fprintf(csvFile, "Compute Throughput,%.2f GFLOPS,%.2f GFLOPS,%s,%.3f\n",
            cResult.gflops, 400.0f,
            cResult.passed ? "PASS" : "FAIL",
            cResult.duration_ms);
        
        fclose(csvFile);
        printf("\nReport saved to results/diagnostic_report.csv\n");
    } else {
        printf("\nWarning: Could not write report to results folder.\n");
    }

    return 0;
}
