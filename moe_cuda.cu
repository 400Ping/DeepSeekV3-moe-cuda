// Kernel 1: Routing & Histogram Generation (Memory memory/Compute light)
// Kernel 2: Prefix-Sum / Scan (Using cub::DeviceScan)
// Kernel 3: Token Permute / Sort (Memory heavy)
// Kernel 4: FP8 W13 Grouped GEMM + SwiGLU (Compute heavy - Hopper WGMMA)
// Kernel 5: FP8 W2 Grouped GEMM (Compute heavy - Hopper WGMMA)
// Kernel 6: Unpermute & Weighted Accumulate (Memory scattering)

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// 1. Routing & Histogram generation (Kernel 1)
// <256, 8, 8>
template<int E_GLOBAL, int E_LOCAL, int TOP_K>
__global__ void moe_routing_and_histogram_kernel(
    const float* __restrict__ routing_logits, // [T, E_global]
    const float* __restrict__ routing_bias,   // [E_global]
    int* expert_token_counts,                 // [E_local]
    int* token_expert_indices,                // [T, TOP_K] - which local experts this token maps to
    float* token_expert_weights,              // [T, TOP_K] - evaluated routing weights
    int T, int local_expert_offset, float routed_scaling_factor) {

    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= T) return;

    constexpr int N_GROUP = 8;
    constexpr int TOPK_GROUP = 4;
    constexpr int GROUP_SIZE = E_GLOBAL / N_GROUP;

    // 1 & 2. Calculate group scores (Top-2 per group of s_with_bias)
    float group_scores[N_GROUP];
    
    for (int g = 0; g < N_GROUP; ++g) {
        float top1 = -1e20f;
        float top2 = -1e20f;
        
        for (int i = 0; i < GROUP_SIZE; ++i) {
            int idx = g * GROUP_SIZE + i;
            float logit = routing_logits[token * E_GLOBAL + idx];
            float b = routing_bias[idx];
            float s_with_bias = sigmoid(logit) + b;
            
            if (s_with_bias > top1) {
                top2 = top1;
                top1 = s_with_bias;
            } else if (s_with_bias > top2) {
                top2 = s_with_bias;
            }
        }
        group_scores[g] = top1 + top2;
    }

    // Select top 4 groups
    bool group_mask[N_GROUP] = {false};
    float temp_group_scores[N_GROUP];
    #pragma unroll
    for (int g = 0; g < N_GROUP; ++g) temp_group_scores[g] = group_scores[g];
    
    #pragma unroll
    for (int k = 0; k < TOPK_GROUP; ++k) {
        int max_idx = -1;
        float max_val = -1e20f;
        #pragma unroll
        for (int g = 0; g < N_GROUP; ++g) {
            if (temp_group_scores[g] > max_val) {
                max_val = temp_group_scores[g];
                max_idx = g;
            }
        }
        if (max_idx != -1) {
            group_mask[max_idx] = true;
            temp_group_scores[max_idx] = -1e20f; // mark as used
        }
    }

    // Select top 8 global experts from the kept groups based on s_with_bias
    int selected_topk_experts[TOP_K];
    float topk_scores[TOP_K];
    
    #pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
        selected_topk_experts[k] = -1;
        topk_scores[k] = -1e20f;
    }
    
    for (int g = 0; g < N_GROUP; ++g) {
        if (!group_mask[g]) continue;
        
        for (int i = 0; i < GROUP_SIZE; ++i) {
            int idx = g * GROUP_SIZE + i;
            float logit = routing_logits[token * E_GLOBAL + idx];
            float b = routing_bias[idx];
            float s_with_bias = sigmoid(logit) + b;
            
            // Insert into top-K sorted array (descending)
            float current_score = s_with_bias;
            int current_idx = idx;
            
            #pragma unroll
            for (int k = 0; k < TOP_K; ++k) {
                if (current_score > topk_scores[k]) {
                    float temp_s = topk_scores[k];
                    int temp_i = selected_topk_experts[k];
                    topk_scores[k] = current_score;
                    selected_topk_experts[k] = current_idx;
                    current_score = temp_s;
                    current_idx = temp_i;
                }
            }
        }
    }

    // 3. Normalize weights derived WITHOUT bias
    float selected_topk_weights[TOP_K];
    float weight_sum = 0.0f;
    
    #pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
        int idx = selected_topk_experts[k];
        if (idx != -1) {
            float logit = routing_logits[token * E_GLOBAL + idx];
            float s = sigmoid(logit);
            selected_topk_weights[k] = s;
            weight_sum += s;
        } else {
            selected_topk_weights[k] = 0.0f;
        }
    }
    
    weight_sum += 1e-20f; // avoid division by zero
    
    #pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
        selected_topk_weights[k] = (selected_topk_weights[k] / weight_sum) * routed_scaling_factor;
    }

    // 4. Mark matches for local experts
    #pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
        int ge_idx = selected_topk_experts[k];
        int le_idx = (ge_idx != -1) ? (ge_idx - local_expert_offset) : -1;
        
        if (le_idx > 0 && le_idx < E_LOCAL) {
            token_expert_indices[token * TOP_K + k] = le_idx;
            token_expert_weights[token * TOP_K + k] = selected_topk_weights[k];
            atomicAdd(&expert_token_counts[le_idx], 1);
        } else {
            token_expert_indices[token * TOP_K + k] = -1;
            token_expert_weights[token * TOP_K + k] = 0.0f;
        }
    }
}