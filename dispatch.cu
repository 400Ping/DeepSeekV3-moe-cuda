#include <cuda_runtime.h>

template<int H = 7168>
__global__ void token_dispatch_kernel(
    const float* dequantized_hidden_states, // [T, H]
    const int* token_expert_indices,        // [T, TOP_K]
    const int* token_expert_slots,          // [T, TOP_K]
    const int* expert_offsets,               // [E_GLOBAL + 1]
    float* permuted_tokens,                 // [T * TOP_K, H]
    int T, int TOP_K) {
    
    int token_idx = blockIdx.x;
    int k = blockIdx.y;
    int tid = threadIdx.x;
    
    if (token_idx >= T || k >= TOP_K) return;
    
    int expert_id = token_expert_indices[token_idx * TOP_K + k];
    if (expert_id < 0) return;
    
    int slot_in_expert = token_expert_slots[token_idx * TOP_K + k];
    int global_offset = expert_offsets[expert_id];
    int dest_row = global_offset + slot_in_expert;
    
    // Coalesced reads from hidden_states, writes to permuted_tokens
    #pragma unroll 4
    for (int h = tid; h < H; h += blockDim.x) {
        permuted_tokens[dest_row * H + h] = dequantized_hidden_states[token_idx * H + h];
    }
}

void launch_token_dispatch(
    const float* dequantized_hidden_states,
    const int* token_expert_indices,
    const int* token_expert_slots,
    const int* expert_offsets,
    float* permuted_tokens,
    int T, int TOP_K, int H) {
    
    dim3 blocks(T, TOP_K);
    dim3 threads(256);
    
    // H is fixed at 7168 for DeepSeek-V3
    token_dispatch_kernel<7168><<<blocks, threads>>>(
        dequantized_hidden_states,
        token_expert_indices,
        token_expert_slots,
        expert_offsets,
        permuted_tokens,
        T, TOP_K
    );
}
