// Author:  justin
// Description:  Token dispatching with FP8 block-scale dequantization
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

// Convert FP8 E4M3 byte → float
__device__ __forceinline__ float fp8e4m3_to_float(__nv_fp8_storage_t val) {
    __half_raw hr = __nv_cvt_fp8_to_halfraw(val, __NV_E4M3);
    return __half2float(*reinterpret_cast<const __half*>(&hr));
}

// Dispatch kernel: scatter tokens into expert-grouped layout with FP8 dequantization
// Grid:  (T, TOP_K)
// Block: (256)
//
// Scale layout: [H/BLOCK_SCALE, T]  (column-major in the block dimension)
//   scale for token t, hidden dim h  =  hidden_states_scale[(h / BLOCK_SCALE) * T + t]
template<int H = 7168, int BLOCK_SCALE = 128>
__global__ void token_dispatch_kernel(
    const __nv_fp8_storage_t* __restrict__ hidden_states_fp8,  // [T, H]  FP8 E4M3
    const float* __restrict__ hidden_states_scale,             // [H/BLOCK_SCALE, T]  float32
    const int* __restrict__ token_expert_indices,              // [T, TOP_K]
    const int* __restrict__ token_expert_slots,                // [T, TOP_K]
    const int* __restrict__ expert_offsets,                    // [E_GLOBAL + 1]
    float* permuted_tokens,                                    // [total_assigned, H]
    int T, int TOP_K) {

    int token_idx = blockIdx.x;
    int k = blockIdx.y;
    int tid = threadIdx.x;

    if (token_idx >= T || k >= TOP_K) return;

    int expert_id = token_expert_indices[token_idx * TOP_K + k];
    if (expert_id < 0) return;

    int slot_in_expert = token_expert_slots[token_idx * TOP_K + k];
    int dest_row = expert_offsets[expert_id] + slot_in_expert;

    const __nv_fp8_storage_t* src_row = hidden_states_fp8 + token_idx * H;
    float* dst_row = permuted_tokens + dest_row * H;

    // Dequantize & scatter: read FP8 + block-scale → write float32
    // Process 4 FP8 values at a time for better memory throughput
    constexpr int VEC_SIZE = 4;
    int h_vec_end = (H / VEC_SIZE) * VEC_SIZE;

    for (int h = tid * VEC_SIZE; h < h_vec_end; h += blockDim.x * VEC_SIZE) {
        int block_idx = h / BLOCK_SCALE;
        float scale = hidden_states_scale[block_idx * T + token_idx];

        // Load 4 FP8 bytes at once
        uint32_t packed = *reinterpret_cast<const uint32_t*>(&src_row[h]);
        __nv_fp8_storage_t b0 = static_cast<__nv_fp8_storage_t>(packed & 0xFF);
        __nv_fp8_storage_t b1 = static_cast<__nv_fp8_storage_t>((packed >> 8) & 0xFF);
        __nv_fp8_storage_t b2 = static_cast<__nv_fp8_storage_t>((packed >> 16) & 0xFF);
        __nv_fp8_storage_t b3 = static_cast<__nv_fp8_storage_t>((packed >> 24) & 0xFF);

        // Check if all 4 values share the same scale block (128-byte aligned blocks)
        // For H=7168 and BLOCK_SCALE=128, blocks are exactly aligned with VEC_SIZE=4
        // so all 4 values within a single iteration share the same scale
        // (only need to check cross-block boundary at h % 128 >= 124, but VEC_SIZE=4 divides 128)

        float f0 = fp8e4m3_to_float(b0) * scale;
        float f1 = fp8e4m3_to_float(b1) * scale;
        float f2 = fp8e4m3_to_float(b2) * scale;
        float f3 = fp8e4m3_to_float(b3) * scale;

        // Vectorized store
        *reinterpret_cast<float4*>(&dst_row[h]) = make_float4(f0, f1, f2, f3);
    }

    // Handle remainder (H=7168 is divisible by 4, so this won't execute)
    for (int h = h_vec_end + tid; h < H; h += blockDim.x) {
        int block_idx = h / BLOCK_SCALE;
        float scale = hidden_states_scale[block_idx * T + token_idx];
        dst_row[h] = fp8e4m3_to_float(src_row[h]) * scale;
    }
}

void launch_token_dispatch(
    const __nv_fp8_storage_t* hidden_states_fp8,
    const float* hidden_states_scale,
    const int* token_expert_indices,
    const int* token_expert_slots,
    const int* expert_offsets,
    float* permuted_tokens,
    int T, int TOP_K, int H) {

    dim3 blocks(T, TOP_K);
    dim3 threads(256);

    token_dispatch_kernel<7168, 128><<<blocks, threads>>>(
        hidden_states_fp8,
        hidden_states_scale,
        token_expert_indices,
        token_expert_slots,
        expert_offsets,
        permuted_tokens,
        T, TOP_K
    );
}
