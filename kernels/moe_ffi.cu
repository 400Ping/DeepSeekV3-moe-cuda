#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/container/tensor.h>

// Include the existing CUDA kernel implementations
#include "moe_routing.cu"
#include "moe_scan.cu"
#include "moe_dispatch.cu"

namespace ffi = tvm::ffi;

// ─── Router FFI ───────────────────────────────────────────────────────────────

void router_ffi_wrapper(ffi::Tensor routing_logits,         // [T, 256]
                        ffi::Tensor routing_bias,           // [256]
                        ffi::Tensor expert_token_counts,    // [256]
                        ffi::Tensor token_expert_indices,   // [T, 8]
                        ffi::Tensor token_expert_weights,   // [T, 8]
                        ffi::Tensor token_expert_slots,     // [T, 8]
                        int T, int local_expert_offset, float routed_scaling_factor) {
    
    const int E_GLOBAL = 256;
    const int E_LOCAL = 32;
    const int TOP_K = 8;
    
    dim3 threads(256);
    dim3 blocks(T);
    
    router<E_GLOBAL, E_LOCAL, TOP_K><<<blocks, threads>>>(
        static_cast<const float*>(routing_logits.data_ptr()),
        static_cast<const __nv_bfloat16*>(routing_bias.data_ptr()),
        static_cast<int*>(expert_token_counts.data_ptr()),
        static_cast<int*>(token_expert_indices.data_ptr()),
        static_cast<float*>(token_expert_weights.data_ptr()),
        static_cast<int*>(token_expert_slots.data_ptr()),
        T, local_expert_offset, routed_scaling_factor
    );
}

static auto _router = ffi::reflection::GlobalDef().def("router_ffi", router_ffi_wrapper);

// ─── Scan (Prefix Sum) FFI ───────────────────────────────────────────────────

void scan_ffi_wrapper(ffi::Tensor input,    // [N]
                      ffi::Tensor output,   // [N]
                      int N) {
    exclusive_scan_cub(
        static_cast<int*>(input.data_ptr()),
        static_cast<int*>(output.data_ptr()),
        N
    );
}

static auto _scan = ffi::reflection::GlobalDef().def("scan_ffi", scan_ffi_wrapper);

// ─── Dispatch (Permutation) FFI ──────────────────────────────────────────────

void dispatch_ffi_wrapper(ffi::Tensor hidden_states_fp8,     // [T, H]  FP8 E4M3
                          ffi::Tensor hidden_states_scale,   // [H/128, T]  float32
                          ffi::Tensor token_expert_indices,  // [T, TOP_K]
                          ffi::Tensor token_expert_slots,    // [T, TOP_K]
                          ffi::Tensor expert_offsets,        // [E_GLOBAL + 1]
                          ffi::Tensor permuted_tokens,       // [total, H]
                          int T, int TOP_K, int H) {
    launch_token_dispatch(
        static_cast<const __nv_fp8_storage_t*>(hidden_states_fp8.data_ptr()),
        static_cast<const float*>(hidden_states_scale.data_ptr()),
        static_cast<const int*>(token_expert_indices.data_ptr()),
        static_cast<const int*>(token_expert_slots.data_ptr()),
        static_cast<const int*>(expert_offsets.data_ptr()),
        static_cast<float*>(permuted_tokens.data_ptr()),
        T, TOP_K, H
    );
}

static auto _dispatch = ffi::reflection::GlobalDef().def("dispatch_ffi", dispatch_ffi_wrapper);
