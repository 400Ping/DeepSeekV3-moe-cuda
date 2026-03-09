__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __inline__ void warp_top2(float val, float &top1, float &top2) {
    top1 = val;
    top2 = -1e20f;

    for (int mask = 16; mask > 0; mask >>= 1) {
        float remote_top1 = __shfl_xor_sync(0xffffffff, top1, mask);
        float remote_top2 = __shfl_xor_sync(0xffffffff, top2, mask);

        if (remote_top1 > top1) {
            top2 = fmaxf(top1, remote_top2);
            top1 = remote_top1;
        } else {
            top2 = fmaxf(top2, remote_top1);
        }
    }
}

struct rankItem {
    float score;
    int idx;
};

__device__ __inline__ void insert_if_higher(rankItem *top8, float score, int idx) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (score > top8[i].score) {
            #pragma unroll
            for (int j = 7; j > i; --j) {
                top8[j] = top8[j - 1];
            }
            top8[i].score = score;
            top8[i].idx = idx;
            break;
        }
    }
}

__device__ __inline__ void warp_top8(rankItem *top8) {

    for (int mask = 16; mask > 0; mask >>= 1) {
        rankItem remote_top8[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            remote_top8[i].score = __shfl_xor_sync(0xffffffff, top8[i].score, mask);
            remote_top8[i].idx = __shfl_xor_sync(0xffffffff, top8[i].idx, mask);
        }
        
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (remote_top8[i].score > -1e10f) {
                insert_if_higher(top8, remote_top8[i].score, remote_top8[i].idx);
            }
        }
    }
}

// blockNum = Token number
// threadNum = group_size * N_GROUP = 32 * 8 = 256

template<int E_GLOBAL, int E_LOCAL, int TOP_K>
__global__ void router(
    const float* __restrict__ routing_logits, // [T, E_global]
    const float* __restrict__ routing_bias,   // [E_global]
    int* expert_token_counts,                 // [E_local]
    int* token_expert_indices,                // [T, TOP_K] - which local experts this token maps to
    float* token_expert_weights,              // [T, TOP_K] - evaluated routing weights
    int T, int local_expert_offset, float routed_scaling_factor) {

    int token = blockIdx.x;
    if (token >= T) return;

    constexpr int N_GROUP = 8;
    constexpr int TOPK_GROUP = 4;
    constexpr int GROUP_SIZE = E_GLOBAL / N_GROUP;

    int tid = threadIdx.x;
    int group_id = tid / GROUP_SIZE;
    int lane_id = tid % GROUP_SIZE;
    
    // 1 & 2. Calculate group scores (Top-2 per group of s_with_bias)
    __shared__ float s_group_scores[N_GROUP];

    float logit = routing_logits[token * E_GLOBAL + tid];
    float s_wb = sigmoid(logit) + routing_bias[tid];

    float g_top1, g_top2;
    warp_top2(s_wb, g_top1, g_top2);

    if (lane_id == 0) {
        s_group_scores[group_id] = g_top1 + g_top2;
    }
    __syncthreads();

    // Top-4 groups
    __shared__ bool group_mask[N_GROUP];

    if (group_id == 0 && lane_id < N_GROUP) {
        float my_group_score = s_group_scores[lane_id];
        int rank = 0;

        #pragma unroll
        for (int g = 0; g < N_GROUP; ++g) {
            float other_score = s_group_scores[g];

            if (other_score > my_group_score || (other_score == my_group_score && g < lane_id)) {
                rank++;
            }
        }

        group_mask[lane_id] = rank < TOPK_GROUP;
    }
    __syncthreads();

    float final_s_wb = group_mask[group_id] ? s_wb : -1e20f;
    
    rankItem val;
    val.score = final_s_wb;
    val.idx = tid;
    
    rankItem local_top8[8];
    warp_top8(val, local_top8);

    // Block-level merge
    __shared__ rankItem shared_top8[N_GROUP * 8];
    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            shared_top8[group_id * 8 + i] = local_top8[i];
        }
    }
    __syncthreads();

    if (tid < 32) {
        rankItem threadCandidates[2];
        threadCandidates[0] = shared_top8[tid];
        threadCandidates[1] = shared_top8[tid + 32];

        rankItem threadBest = (threadCandidates[0].score > threadCandidates[1].score) ? threadCandidates[0] : threadCandidates[1];
        
        rankItem thread_top8[8];
        warp_top8(threadBest, thread_top8);
        
        if (lane_id == 0) {
            float selected_topk_weights[8];
            float weight_sum = 0.0f;

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int idx = thread_top8[i].idx;
                if (idx != -1) {
                    float logit = routing_logits[token * E_GLOBAL + idx];
                    float s = sigmoid(logit);
                    selected_topk_weights[i] = s;
                    weight_sum += s;
                } else {
                    selected_topk_weights[i] = 0.0f;
                }
            }

            weight_sum += 1e-20f;

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int ge_idx = thread_top8[i].idx;
                if (ge_idx != -1) {
                    token_expert_indices[token * TOP_K + i] = ge_idx;
                    token_expert_weights[token * TOP_K + i] = (selected_topk_weights[i] / weight_sum) * routed_scaling_factor;
                    
                    int le_idx = ge_idx - local_expert_offset;
                    if (le_idx >= 0 && le_idx < E_LOCAL) {
                        atomicAdd(&expert_token_counts[le_idx], 1);
                    }
                }
            }
        }
    }
}