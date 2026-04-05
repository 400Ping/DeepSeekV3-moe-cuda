// Standalone test for Kernel 2: Prefix Sum (Scan)
// Compile:  nvcc -arch=sm_89 -o test_scan test_scan.cu
// Run:      ./test_scan

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "scan.cu"

void cpu_exclusive_scan(const int* in, int* out, int n) {
    out[0] = 0;
    for (int i = 1; i < n; i++) out[i] = out[i-1] + in[i-1];
}

bool run_test(const int* h_input, int n, const char* name) {
    int* h_expected = new int[n];
    int* h_output   = new int[n];
    cpu_exclusive_scan(h_input, h_expected, n);

    int *d_in, *d_out;
    cudaMalloc(&d_in,  n * sizeof(int));
    cudaMalloc(&d_out, n * sizeof(int));
    cudaMemcpy(d_in, h_input, n * sizeof(int), cudaMemcpyHostToDevice);

    exclusive_scan_cub(d_in, d_out, n);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[FAIL] %s — CUDA error: %s\n", name, cudaGetErrorString(err));
        cudaFree(d_in); cudaFree(d_out);
        delete[] h_expected; delete[] h_output;
        return false;
    }

    cudaMemcpy(h_output, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);

    bool pass = (memcmp(h_expected, h_output, n * sizeof(int)) == 0);
    printf("[%s] %s  (n=%d)\n", pass ? "PASS" : "FAIL", name, n);
    if (!pass) {
        for (int i = 0; i < n && i < 20; i++)
            if (h_expected[i] != h_output[i])
                printf("       idx %d: expected %d, got %d\n", i, h_expected[i], h_output[i]);
    }

    cudaFree(d_in); cudaFree(d_out);
    delete[] h_expected; delete[] h_output;
    return pass;
}

int main() {
    printf("=== Kernel 2: Prefix Sum (Scan) Tests ===\n\n");
    int pass = 0, total = 0;

    // Test 1: Basic
    { int in[] = {3,0,5,2,0}; total++; pass += run_test(in, 5, "Basic [3,0,5,2,0]"); }

    // Test 2: All zeros
    { int in[] = {0,0,0,0,0}; total++; pass += run_test(in, 5, "All zeros"); }

    // Test 3: Single expert has all tokens
    { int in[] = {0,0,100,0,0}; total++; pass += run_test(in, 5, "Single expert"); }

    // Test 4: DeepSeek-V3 realistic (E=256, scan 257 elements)
    {
        const int N = 257;
        int in[N];
        srand(42);
        for (int i = 0; i < 256; i++) in[i] = rand() % 200;
        in[256] = 0;
        total++; pass += run_test(in, N, "DeepSeek-V3 (257 elements)");
    }

    // Test 5: Verify last element = total sum
    {
        int in[] = {3,7,2,8,0};
        int *d_in, *d_out;
        cudaMalloc(&d_in, 5*sizeof(int));
        cudaMalloc(&d_out, 5*sizeof(int));
        cudaMemcpy(d_in, in, 5*sizeof(int), cudaMemcpyHostToDevice);
        exclusive_scan_cub(d_in, d_out, 5);
        cudaDeviceSynchronize();
        int last; cudaMemcpy(&last, d_out+4, sizeof(int), cudaMemcpyDeviceToHost);
        bool ok = (last == 20);
        printf("[%s] Last element = total sum (%d == 20)\n", ok?"PASS":"FAIL", last);
        total++; pass += ok;
        cudaFree(d_in); cudaFree(d_out);
    }

    printf("\n=== %d / %d passed %s===\n", pass, total, pass==total ? "✓ " : "");
    return (pass == total) ? 0 : 1;
}
