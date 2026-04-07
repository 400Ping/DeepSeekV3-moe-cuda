#include "kernel4.cuh"

#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

using namespace moe_spec;

static float silu_f(float x) {
    return x / (1.f + expf(-x));
}

void k4_reference_cpu(
    const float* hidden_states_f32,
    const float* hidden_states_scale,
    const int*   token_indices,
    const int*   expert_token_offsets,
    const float* gemm1_weights_f32,
    const float* gemm1_weights_scale,
    const float* gemm2_weights_f32,
    const float* gemm2_weights_scale,
    const float* token_expert_weights,
    float        routed_scaling_factor,
    int          seq_len,
    int          total_dispatched_tokens,
    float*       output_f32)
{
    memset(output_f32, 0, (size_t)seq_len * HIDDEN_SIZE * sizeof(float));

    for (int tok = 0; tok < total_dispatched_tokens; ++tok) {
        int expert_id = 0;
        for (int e = 0; e < NUM_LOCAL_EXPERTS; ++e) {
            if (tok >= expert_token_offsets[e] && tok < expert_token_offsets[e + 1]) {
                expert_id = e;
                break;
            }
        }

        int orig_tok = token_indices[tok];
        assert(orig_tok >= 0 && orig_tok < seq_len);

        const float* W1 = gemm1_weights_f32 + (size_t)expert_id * GEMM1_OUT_SIZE * HIDDEN_SIZE;
        const float* W1s = gemm1_weights_scale + (size_t)expert_id * NUM_GEMM1_OUT_BLOCKS * NUM_HIDDEN_BLOCKS;
        const float* W2 = gemm2_weights_f32 + (size_t)expert_id * HIDDEN_SIZE * INTERMEDIATE_SIZE;
        const float* W2s = gemm2_weights_scale + (size_t)expert_id * NUM_HIDDEN_BLOCKS * NUM_INTER_BLOCKS;
        float rw = token_expert_weights[tok] * routed_scaling_factor;

        std::vector<float> inter(INTERMEDIATE_SIZE, 0.f);
        for (int out_col = 0; out_col < INTERMEDIATE_SIZE; ++out_col) {
            float acc_up = 0.f;
            float acc_gate = 0.f;
            int bn_up = out_col / BLOCK_SIZE;
            int bn_gate = (INTERMEDIATE_SIZE + out_col) / BLOCK_SIZE;
            for (int bk = 0; bk < NUM_HIDDEN_BLOCKS; ++bk) {
                float a_scale = hidden_states_scale[bk * seq_len + orig_tok];
                float ws_up = W1s[bn_up * NUM_HIDDEN_BLOCKS + bk];
                float ws_gate = W1s[bn_gate * NUM_HIDDEN_BLOCKS + bk];

                float dot_up = 0.f;
                float dot_gate = 0.f;
                int k_base = bk * BLOCK_SIZE;
                for (int k = 0; k < BLOCK_SIZE; ++k) {
                    float a = hidden_states_f32[tok * HIDDEN_SIZE + k_base + k] * a_scale;
                    dot_up += a * W1[out_col * HIDDEN_SIZE + k_base + k] * ws_up;
                    dot_gate += a * W1[(INTERMEDIATE_SIZE + out_col) * HIDDEN_SIZE + k_base + k] * ws_gate;
                }

                acc_up += dot_up;
                acc_gate += dot_gate;
            }

            inter[out_col] = acc_gate * silu_f(acc_up);
        }

        for (int out_h = 0; out_h < HIDDEN_SIZE; ++out_h) {
            float acc = 0.f;
            int hb = out_h / BLOCK_SIZE;
            for (int ib = 0; ib < NUM_INTER_BLOCKS; ++ib) {
                float tile_scale = W2s[hb * NUM_INTER_BLOCKS + ib];
                int k_base = ib * BLOCK_SIZE;
                float dot = 0.f;
                for (int k = 0; k < BLOCK_SIZE; ++k) {
                    float a = inter[k_base + k];
                    float w = W2[out_h * INTERMEDIATE_SIZE + k_base + k];
                    dot += a * w;
                }
                acc += dot * tile_scale;
            }

            output_f32[orig_tok * HIDDEN_SIZE + out_h] += acc * rw;
        }
    }
}
