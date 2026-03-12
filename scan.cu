#include <cub/cub.cuh>
#include <cuda_runtime.h>

void exclusive_scan_cub(int* d_input, int* d_output, int num_items) {
    // Determine temporary device storage requirements
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, num_items);
    
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    // Run exclusive-sum scan
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, num_items);
    
    // Cleanup
    cudaFree(d_temp_storage);
}
