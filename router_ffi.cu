#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/container/tensor.h>

// Include CUDA kernel implementations
#include "router_v2.cu"
#include "scan.cu"
#include "dispatch.cu"

namespace ffi = tvm::ffi;

// ============================================================================
// Kernel 1: Router FFI Wrapper
// ============================================================================
void router_ffi_wrapper(ffi::Tensor routing_logits,         // [T, 256]
                        ffi::Tensor routing_bias,           // [256]
                        ffi::Tensor expert_token_counts,    // [256] or [E_local]
                        ffi::Tensor token_expert_indices,   // [T, 8]
                        ffi::Tensor token_expert_weights,    // [T, 8]
                        int T, int local_expert_offset, float routed_scaling_factor) {
    
    // DeepSeek-V3 routing constants
    const int E_GLOBAL = 256;
    const int E_LOCAL = 32; // This matches the de-facto DeepSeek-V3 config
    const int TOP_K = 8;
    
    // Thread block configuration: one block per token, 256 threads per block
    // (256 threads = 8 groups * 32 experts/group)
    dim3 threads(256);
    dim3 blocks(T);
    
    // Launch the kernel
    // Note: Ensure the expert_token_counts buffer is large enough (256) 
    // to avoid overflow if the kernel doesn't mask out non-local experts.
    router<E_GLOBAL, E_LOCAL, TOP_K><<<blocks, threads>>>(
        static_cast<const float*>(routing_logits.data_ptr()),
        static_cast<const float*>(routing_bias.data_ptr()),
        static_cast<int*>(expert_token_counts.data_ptr()),
        static_cast<int*>(token_expert_indices.data_ptr()),
        static_cast<float*>(token_expert_weights.data_ptr()),
        T, local_expert_offset, routed_scaling_factor
    );
}

// ============================================================================
// Kernel 2: Prefix Sum (Scan) FFI Wrapper
// ============================================================================
// Converts expert_token_counts[E] into expert_offsets[E+1] via exclusive scan.
// After this kernel:
//   expert_offsets[e] = starting index in the dispatch buffer for expert e
//   expert_offsets[E] = total number of dispatched tokens
void scan_ffi_wrapper(ffi::Tensor d_input,   // [num_items] expert token counts
                      ffi::Tensor d_output,  // [num_items] expert offsets (exclusive scan)
                      int num_items) {
    exclusive_scan_cub(
        static_cast<int*>(d_input.data_ptr()),
        static_cast<int*>(d_output.data_ptr()),
        num_items
    );
}

// ============================================================================
// Kernel 3: Token Dispatch FFI Wrapper
// ============================================================================
void dispatch_ffi_wrapper(ffi::Tensor dequantized_hidden_states,  // [T, H]
                          ffi::Tensor token_expert_indices,        // [T, TOP_K]
                          ffi::Tensor token_expert_slots,          // [T, TOP_K]
                          ffi::Tensor expert_offsets,               // [E_GLOBAL + 1]
                          ffi::Tensor permuted_tokens,             // [T * TOP_K, H]
                          int T, int TOP_K, int H) {
    launch_token_dispatch(
        static_cast<const float*>(dequantized_hidden_states.data_ptr()),
        static_cast<const int*>(token_expert_indices.data_ptr()),
        static_cast<const int*>(token_expert_slots.data_ptr()),
        static_cast<const int*>(expert_offsets.data_ptr()),
        static_cast<float*>(permuted_tokens.data_ptr()),
        T, TOP_K, H
    );
}

// ============================================================================
// TVM-FFI Global Function Registrations
// ============================================================================
static auto _router   = ffi::reflection::GlobalDef().def("router_ffi",   router_ffi_wrapper);
static auto _scan     = ffi::reflection::GlobalDef().def("scan_ffi",     scan_ffi_wrapper);
static auto _dispatch = ffi::reflection::GlobalDef().def("dispatch_ffi", dispatch_ffi_wrapper);
