import torch
import tvm_ffi
import pytest
import os

lib_path = os.path.abspath("librouter_ffi.so")
tvm_ffi.load_module(lib_path)

E_GLOBAL = 256
TOP_K    = 8
H        = 7168
BLOCK_SCALE = 128
NUM_HIDDEN_BLOCKS = H // BLOCK_SCALE  # 56


def dispatch_reference(hidden_states_fp8, hidden_states_scale,
                       token_expert_indices, expert_offsets, token_expert_slots,
                       T, total_assigned):
    hs_f32 = hidden_states_fp8.to(torch.float32)
    scale = hidden_states_scale.to(torch.float32)              # [56, T]

    scale_t = scale.permute(1, 0).contiguous()                 # [T, 56]
    scale_expanded = (
        scale_t.unsqueeze(-1)
        .expand(T, NUM_HIDDEN_BLOCKS, BLOCK_SCALE)
        .reshape(T, H)
        .contiguous()
    )
    dequantized = hs_f32 * scale_expanded                      # [T, H]

    permuted_tokens = torch.zeros(total_assigned, H, dtype=torch.float32,
                                  device=hidden_states_fp8.device)

    indices = token_expert_indices.cpu()
    slots   = token_expert_slots.cpu()
    offsets = expert_offsets.cpu()

    for t in range(T):
        for k in range(TOP_K):
            expert_id = indices[t, k].item()
            if expert_id < 0:
                continue
            slot = slots[t, k].item()
            dest_row = offsets[expert_id].item() + slot
            permuted_tokens[dest_row] = dequantized[t]

    return permuted_tokens


def build_dispatch_inputs(T, seed=42):
    torch.manual_seed(seed)

    hidden_states_fp8 = torch.randn(T, H, device='cuda', dtype=torch.float32) \
                             .to(torch.float8_e4m3fn)
    hidden_states_scale = torch.rand(NUM_HIDDEN_BLOCKS, T, device='cuda',
                                     dtype=torch.float32) + 0.1

    # Simulate routing: each token picks TOP_K=8 distinct experts
    token_expert_indices = torch.zeros(T, TOP_K, device='cuda', dtype=torch.int32)
    for t in range(T):
        perm = torch.randperm(E_GLOBAL)[:TOP_K].sort().values
        token_expert_indices[t] = perm.to(torch.int32)

    # Count how many tokens each expert got
    counts = torch.zeros(E_GLOBAL, dtype=torch.int32)
    indices_cpu = token_expert_indices.cpu()
    for t in range(T):
        for k in range(TOP_K):
            eid = indices_cpu[t, k].item()
            counts[eid] += 1

    # Assign slots
    token_expert_slots = torch.zeros(T, TOP_K, device='cuda', dtype=torch.int32)
    slot_counter = torch.zeros(E_GLOBAL, dtype=torch.int32)
    for t in range(T):
        for k in range(TOP_K):
            eid = indices_cpu[t, k].item()
            token_expert_slots[t, k] = slot_counter[eid].item()
            slot_counter[eid] += 1

    # Exclusive prefix sum for offsets
    expert_offsets = torch.zeros(E_GLOBAL + 1, dtype=torch.int32)
    expert_offsets[1:] = torch.cumsum(counts, dim=0)
    expert_offsets = expert_offsets.to(device='cuda')

    total_assigned = expert_offsets[E_GLOBAL].item()

    return (hidden_states_fp8, hidden_states_scale,
            token_expert_indices, token_expert_slots,
            expert_offsets, total_assigned)


@pytest.mark.parametrize("T", [1, 4, 8, 32])
@pytest.mark.parametrize("seed", [42, 123])
def test_dispatch_kernel(T, seed):
    (hidden_states_fp8, hidden_states_scale,
     token_expert_indices, token_expert_slots,
     expert_offsets, total_assigned) = build_dispatch_inputs(T, seed)

    # Run kernel 3
    permuted_tokens = torch.zeros(total_assigned, H, device='cuda', dtype=torch.float32)
    dispatch_func = tvm_ffi.get_global_func("dispatch_ffi")
    dispatch_func(
        tvm_ffi.from_dlpack(hidden_states_fp8.contiguous()),
        tvm_ffi.from_dlpack(hidden_states_scale.contiguous()),
        tvm_ffi.from_dlpack(token_expert_indices.contiguous()),
        tvm_ffi.from_dlpack(token_expert_slots.contiguous()),
        tvm_ffi.from_dlpack(expert_offsets.contiguous()),
        tvm_ffi.from_dlpack(permuted_tokens),
        T, TOP_K, H,
    )
    torch.cuda.synchronize()

    # PyTorch reference
    ref = dispatch_reference(
        hidden_states_fp8, hidden_states_scale,
        token_expert_indices, expert_offsets, token_expert_slots,
        T, total_assigned,
    )

    torch.testing.assert_close(
        permuted_tokens.cpu(), ref.cpu(),
        atol=1e-3, rtol=1e-3,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
