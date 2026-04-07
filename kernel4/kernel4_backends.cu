#include "kernel4_internal.cuh"

namespace kernel4_internal {

cudaError_t launch_fallback_backend(const Kernel4Problem& p,
                                    const Kernel4Workspace& workspace,
                                    int total_tokens) {
    dim3 block(256);
    dim3 gemm1_grid(total_tokens, (INTERMEDIATE_SIZE + block.x - 1) / block.x);
    fp8_gemm1_swiglu_reference_kernel<<<gemm1_grid, block, 0, p.stream>>>(
        p.hidden_states,
        p.hidden_states_scale,
        p.token_indices,
        p.expert_token_offsets,
        p.gemm1_weights,
        p.gemm1_weights_scale,
        workspace.gemm1_output,
        total_tokens,
        p.seq_len,
        NUM_LOCAL_EXPERTS
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemsetAsync(
        workspace.output_accum,
        0,
        output_accum_bytes(p.seq_len),
        p.stream));

    dim3 gemm2_grid(total_tokens, (HIDDEN_SIZE + block.x - 1) / block.x);
    fp8_gemm2_project_kernel<<<gemm2_grid, block, 0, p.stream>>>(
        workspace.gemm1_output,
        p.expert_token_offsets,
        p.gemm2_weights,
        p.gemm2_weights_scale,
        workspace.gemm2_output,
        total_tokens,
        NUM_LOCAL_EXPERTS
    );
    CUDA_CHECK(cudaGetLastError());

    combine_projected_kernel<<<gemm2_grid, block, 0, p.stream>>>(
        workspace.gemm2_output,
        p.token_indices,
        p.token_expert_weights,
        p.routed_scaling_factor,
        workspace.output_accum,
        total_tokens,
        p.seq_len
    );
    CUDA_CHECK(cudaGetLastError());

    int total_output_elems = p.seq_len * HIDDEN_SIZE;
    dim3 pack_grid((total_output_elems + block.x - 1) / block.x);
    f32_to_bf16_kernel<<<pack_grid, block, 0, p.stream>>>(
        workspace.output_accum,
        p.output,
        total_output_elems
    );
    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

cudaError_t launch_tiled_backend(const Kernel4Problem& p,
                                 const Kernel4Workspace& workspace,
                                 int total_tokens) {
    dim3 block_gemm1(BN);
    for (int expert = 0; expert < NUM_LOCAL_EXPERTS; ++expert) {
        int begin = 0;
        int end = 0;
        CUDA_CHECK(cudaMemcpy(&begin,
            p.expert_token_offsets + expert,
            sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&end,
            p.expert_token_offsets + expert + 1,
            sizeof(int), cudaMemcpyDeviceToHost));

        int token_count = end - begin;
        if (token_count <= 0) {
            continue;
        }

        const fp8_e4m3* act_e = p.hidden_states + (size_t)begin * HIDDEN_SIZE;
        const int* token_indices_e = p.token_indices + begin;
        const fp8_e4m3* w_e = p.gemm1_weights +
            (size_t)expert * GEMM1_OUT_SIZE * HIDDEN_SIZE;
        const float* ws_e = p.gemm1_weights_scale +
            (size_t)expert * NUM_GEMM1_OUT_BLOCKS * NUM_HIDDEN_BLOCKS;
        __nv_bfloat16* out_e = workspace.gemm1_output + (size_t)begin * INTERMEDIATE_SIZE;

        dim3 grid_gemm1((token_count + BT - 1) / BT, NUM_INTER_BLOCKS);
        fp8_gemm1_swiglu_kernel<<<grid_gemm1, block_gemm1, 0, p.stream>>>(
            act_e,
            p.hidden_states_scale,
            token_indices_e,
            token_count,
            w_e,
            ws_e,
            out_e,
            p.seq_len
        );
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMemsetAsync(
        workspace.output_accum,
        0,
        output_accum_bytes(p.seq_len),
        p.stream));

    dim3 block(256);
    dim3 gemm2_grid(total_tokens, (HIDDEN_SIZE + block.x - 1) / block.x);
    fp8_gemm2_project_kernel<<<gemm2_grid, block, 0, p.stream>>>(
        workspace.gemm1_output,
        p.expert_token_offsets,
        p.gemm2_weights,
        p.gemm2_weights_scale,
        workspace.gemm2_output,
        total_tokens,
        NUM_LOCAL_EXPERTS
    );
    CUDA_CHECK(cudaGetLastError());

    combine_projected_kernel<<<gemm2_grid, block, 0, p.stream>>>(
        workspace.gemm2_output,
        p.token_indices,
        p.token_expert_weights,
        p.routed_scaling_factor,
        workspace.output_accum,
        total_tokens,
        p.seq_len
    );
    CUDA_CHECK(cudaGetLastError());

    int total_output_elems = p.seq_len * HIDDEN_SIZE;
    dim3 pack_grid((total_output_elems + block.x - 1) / block.x);
    f32_to_bf16_kernel<<<pack_grid, block, 0, p.stream>>>(
        workspace.output_accum,
        p.output,
        total_output_elems
    );
    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

}  // namespace kernel4_internal
