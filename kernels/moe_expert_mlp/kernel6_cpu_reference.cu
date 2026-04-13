#include "kernel6.cuh"

#include <cassert>
#include <cstring>

using namespace moe_spec;

void k6_reference_cpu(
    const __nv_bfloat16* hidden_states_bf16,
    const int*           token_indices,
    const int*           expert_token_offsets,
    const float*         gemm2_weights_f32,
    const float*         gemm2_weights_scale,
    const float*         token_expert_weights,
    float                routed_scaling_factor,
    int                  seq_len,
    int                  total_dispatched_tokens,
    float*               output_f32)
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

        const float* W2 = gemm2_weights_f32 + (size_t)expert_id * HIDDEN_SIZE * INTERMEDIATE_SIZE;
        const float* W2s = gemm2_weights_scale + (size_t)expert_id * NUM_HIDDEN_BLOCKS * NUM_INTER_BLOCKS;
        float rw = token_expert_weights[tok] * routed_scaling_factor;

        for (int out_h = 0; out_h < HIDDEN_SIZE; ++out_h) {
            float acc = 0.f;
            int hb = out_h / BLOCK_SIZE;
            for (int ib = 0; ib < NUM_INTER_BLOCKS; ++ib) {
                float tile_scale = W2s[hb * NUM_INTER_BLOCKS + ib];
                int k_base = ib * BLOCK_SIZE;
                float dot = 0.f;
                for (int k = 0; k < BLOCK_SIZE; ++k) {
                    float a = __bfloat162float(hidden_states_bf16[(size_t)tok * INTERMEDIATE_SIZE + k_base + k]);
                    float w = W2[out_h * INTERMEDIATE_SIZE + k_base + k];
                    dot += a * w;
                }
                acc += dot * tile_scale;
            }

            output_f32[orig_tok * HIDDEN_SIZE + out_h] += acc * rw;
        }
    }
}
