#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/container/tensor.h>

// Include the existing CUDA kernel implementation
#include "test_group_max.cu"

namespace ffi = tvm::ffi;

// Wrapper function that matches the TVM-FFI interface requirements
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

// Register the wrapper function as a TVM global function
// This allows it to be called from Python using tvm_ffi.get_global_func
static auto _ = ffi::reflection::GlobalDef().def("router_ffi", router_ffi_wrapper);
