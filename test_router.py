import torch
import tvm_ffi
import numpy as np
import pytest
import os

# Load the compiled library
lib_path = os.path.abspath("librouter_ffi.so")
tvm_ffi.load_module(lib_path)

def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))

def router_reference(routing_logits, routing_bias, routed_scaling_factor=1.0):
    """
    Reference implementation matching the logic in test_group_max.cu.
    Now implements normalized weights based on sigmoid(logits).
    """
    T, E = routing_logits.shape
    N_GROUP = 8
    TOPK_GROUP = 4
    TOP_K = 8
    
    # Calculate scores with bias for ranking
    s = sigmoid(routing_logits)
    s_with_bias = s + routing_bias
    
    # Grouping
    group_size = E // N_GROUP
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)
    
    # Group scores = sum of top-2 values within each group
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2) # [T, 8]
    
    # Select topk_group groups -> group mask
    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=True)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    
    # Mask out non-selected groups
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E)
    neg_inf = -1e20
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    
    # Global top-k (within kept groups), based on s_with_bias
    topk_scores, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=True)
    
    # Mask out results that were pruned
    valid_mask = topk_scores > -1e10
    
    # Combination weights: use s (without bias) for normalization
    # Construct weights for normalization
    weights = torch.zeros_like(s)
    for b in range(T):
        for k in range(TOP_K):
            idx = topk_idx[b, k]
            if valid_mask[b, k]:
                weights[b, idx] = s[b, idx]
    
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    normalized_weights = (weights / weights_sum) * routed_scaling_factor
    
    # Final top-k results
    final_indices = topk_idx.clone()
    final_indices[~valid_mask] = -1
    
    final_weights = torch.zeros_like(topk_scores)
    for b in range(T):
        for k in range(TOP_K):
            idx = topk_idx[b, k]
            if valid_mask[b, k]:
                final_weights[b, k] = normalized_weights[b, idx]
    
    return final_indices, final_weights

@pytest.mark.parametrize("T", [1, 8, 32])
@pytest.mark.parametrize("scaling_factor", [1.0, 2.5])
def test_router_ffi(T, scaling_factor):
    E_GLOBAL = 256
    TOP_K = 8
    
    # Random input
    torch.manual_seed(42)
    routing_logits = torch.randn(T, E_GLOBAL, device='cuda', dtype=torch.float32)
    routing_bias = torch.randn(E_GLOBAL, device='cuda', dtype=torch.bfloat16)
    
    # Reference results
    ref_indices, ref_scores = router_reference(routing_logits, routing_bias, scaling_factor)

    # FP8 Hidden States & Scales
    H = 7168
    hidden_states = torch.randn(T, H, device='cuda', dtype=torch.float32).to(torch.float8_e4m3fn)
    hidden_states_scale = torch.rand(H // 128, T, device='cuda', dtype=torch.float32)
    dequant_hidden_states = torch.zeros(T, H, device='cuda', dtype=torch.float32)

    # CUDA results buffers
    expert_token_counts = torch.zeros(E_GLOBAL, device='cuda', dtype=torch.int32)
    token_expert_slots = torch.zeros(T, TOP_K, device='cuda', dtype=torch.int32)
    token_expert_indices = torch.full((T, TOP_K), -1, device='cuda', dtype=torch.int32)
    token_expert_weights = torch.zeros(T, TOP_K, device='cuda', dtype=torch.float32)
    
    # Get and call the FFI function
    router_func = tvm_ffi.get_global_func("router_ffi")
    
    # DLPack tensors
    router_func(
        tvm_ffi.from_dlpack(routing_logits.contiguous()),
        tvm_ffi.from_dlpack(routing_bias.contiguous()),
        tvm_ffi.from_dlpack(hidden_states.contiguous()),
        tvm_ffi.from_dlpack(hidden_states_scale.contiguous()),
        tvm_ffi.from_dlpack(dequant_hidden_states),
        tvm_ffi.from_dlpack(token_expert_slots),
        tvm_ffi.from_dlpack(expert_token_counts),
        tvm_ffi.from_dlpack(token_expert_indices),
        tvm_ffi.from_dlpack(token_expert_weights),
        T, 0, scaling_factor
    )
    
    # 1. Verify indices
    torch.testing.assert_close(token_expert_indices, ref_indices.to(torch.int32))
    
    # 2. Verify scores
    torch.testing.assert_close(token_expert_weights, ref_scores, atol=1e-5, rtol=1e-5)

    # 3. Verify Dequantization
    A_fp32 = hidden_states.to(torch.float32)
    A_scale = hidden_states_scale.to(torch.float32) 
    A_scale_expanded = A_scale.repeat_interleave(128, dim=0).t() 
    expected_dequant = A_fp32 * A_scale_expanded
    torch.testing.assert_close(dequant_hidden_states, expected_dequant, atol=1e-5, rtol=1e-5)
    
    # 4. Verify counts
    expected_counts = torch.zeros(E_GLOBAL, device='cuda', dtype=torch.int32)
    valid_mask = ref_indices != -1
    expected_counts.scatter_add_(0, ref_indices[valid_mask].flatten(), torch.ones_like(ref_indices[valid_mask].flatten(), dtype=torch.int32))
    torch.testing.assert_close(expert_token_counts, expected_counts)

    # 5. Verify Scan (Prefix Sum)
    expert_offsets = torch.zeros(E_GLOBAL + 1, device='cuda', dtype=torch.int32)
    scan_func = tvm_ffi.get_global_func("scan_ffi")
    expert_offsets_input = torch.zeros(E_GLOBAL + 1, device='cuda', dtype=torch.int32)
    expert_offsets_input[:E_GLOBAL] = expert_token_counts
    
    scan_func(
        tvm_ffi.from_dlpack(expert_offsets_input),
        tvm_ffi.from_dlpack(expert_offsets),
        E_GLOBAL + 1
    )
    
    expected_offsets = torch.zeros(E_GLOBAL + 1, device='cuda', dtype=torch.int32)
    expected_offsets[1:] = torch.cumsum(expert_token_counts, dim=0)
    torch.testing.assert_close(expert_offsets, expected_offsets)

    # 6. Verify Dispatch (Permutation)
    permuted_tokens = torch.zeros(T * TOP_K, H, device='cuda', dtype=torch.float32)
    dispatch_func = tvm_ffi.get_global_func("dispatch_ffi")
    
    dispatch_func(
        tvm_ffi.from_dlpack(dequant_hidden_states),
        tvm_ffi.from_dlpack(token_expert_indices),
        tvm_ffi.from_dlpack(token_expert_slots),
        tvm_ffi.from_dlpack(expert_offsets),
        tvm_ffi.from_dlpack(permuted_tokens),
        T, TOP_K, H
    )

    # Reference Permutation logic matching deepSeekV3_moe.py
    # We use the SLOTS provided by the kernel to build the reference,
    # because the order of tokens within an expert is non-deterministic (atomicAdd).
    expected_permuted = torch.zeros(T * TOP_K, H, device='cuda', dtype=torch.float32)
    for t in range(T):
        for k in range(TOP_K):
            e = token_expert_indices[t, k].item()
            if e != -1:
                slot = token_expert_slots[t, k].item()
                dest_row = expected_offsets[e].item() + slot
                expected_permuted[dest_row] = dequant_hidden_states[t]

    torch.testing.assert_close(permuted_tokens, expected_permuted, atol=1e-5, rtol=1e-5)

if __name__ == "__main__":
    pytest.main([__file__])
