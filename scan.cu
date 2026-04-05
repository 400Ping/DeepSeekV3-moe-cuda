// Kernel 2: Prefix Sum (Scan) using CUB
// Logic: Converts expert token counts into global offsets (pointers)
//        in the dispatch buffer via exclusive prefix sum.
//
// Input:  expert_token_counts[E]  — number of tokens routed to each expert
// Output: expert_offsets[E+1]     — exclusive prefix sum
//         expert_offsets[e] = sum of counts for experts 0..e-1
//         expert_offsets[E] = total number of routed tokens
//
// Uses cub::DeviceScan::ExclusiveSum which is a highly optimized,
// work-efficient parallel scan implementation that automatically
// selects the best algorithm for the given hardware and problem size.

#include <cub/cub.cuh>
#include <cuda_runtime.h>

// ============================================================================
// Core Scan Implementation
// ============================================================================

/// @brief Performs an exclusive prefix sum (scan) on device memory using CUB.
///
/// Given an input array [c0, c1, c2, ..., cN-1], produces:
///   output[0] = 0
///   output[1] = c0
///   output[2] = c0 + c1
///   ...
///   output[N-1] = c0 + c1 + ... + cN-2
///
/// This is used to convert per-expert token counts into global offsets
/// into the dispatch buffer. The last element output[E] gives the total
/// number of tokens dispatched, which equals the dispatch buffer size.
///
/// @param d_input   Device pointer to input array (expert token counts)
/// @param d_output  Device pointer to output array (expert offsets)
/// @param num_items Number of elements to scan
/// @param stream    CUDA stream for async execution (default: 0)
void exclusive_scan_cub(int* d_input, int* d_output, int num_items,
                        cudaStream_t stream = 0) {
    // Phase 1: Query temporary storage requirements
    // CUB uses a two-phase pattern: first call with nullptr to get size,
    // then allocate and call again to execute.
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes,
        d_input, d_output, num_items, stream);

    // Phase 2: Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Phase 3: Execute the exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes,
        d_input, d_output, num_items, stream);

    // Cleanup temporary storage
    cudaFree(d_temp_storage);
}

// ============================================================================
// Persistent-buffer variant (avoids repeated malloc/free in hot path)
// ============================================================================

/// @brief Scan variant that reuses pre-allocated temporary storage.
///
/// For production use where the scan is called repeatedly (e.g., every
/// inference step), pre-allocating the temporary buffer avoids the
/// overhead of cudaMalloc/cudaFree on each call.
///
/// Usage:
///   1. Call get_scan_temp_storage_bytes() once to get required size
///   2. Allocate buffer once with cudaMalloc
///   3. Call exclusive_scan_cub_persistent() on each iteration
///   4. Free buffer on shutdown
///
/// @param d_temp_storage  Pre-allocated temporary storage
/// @param temp_storage_bytes  Size of temporary storage in bytes
/// @param d_input   Device pointer to input array
/// @param d_output  Device pointer to output array
/// @param num_items Number of elements to scan
/// @param stream    CUDA stream (default: 0)
void exclusive_scan_cub_persistent(void* d_temp_storage,
                                   size_t temp_storage_bytes,
                                   int* d_input, int* d_output,
                                   int num_items,
                                   cudaStream_t stream = 0) {
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes,
        d_input, d_output, num_items, stream);
}

/// @brief Returns the temporary storage size required by CUB for a given
///        scan problem size. Use this to pre-allocate a persistent buffer.
///
/// @param num_items Number of elements to scan
/// @return          Required temporary storage size in bytes
size_t get_scan_temp_storage_bytes(int num_items) {
    void* d_temp = nullptr;
    size_t bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp, bytes,
                                  (int*)nullptr, (int*)nullptr, num_items);
    return bytes;
}
