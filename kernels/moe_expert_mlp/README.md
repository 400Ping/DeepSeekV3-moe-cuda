# Kernel4 / Kernel6

Local MoE expert path for:
- GEMM1 (`W1 || W3`)
- SwiGLU
- GEMM2
- weighted combine back to `[seq_len, hidden_size]`

Standalone stage APIs:
- `kernel4`: GEMM1 + SwiGLU, with the fused end-to-end path still available
- `kernel6`: GEMM2 + routed combine back to sequence order

## Quantization

- Block size: `128`
- `hidden_states_scale[bk, token]`
- `gemm1_weights_scale[expert, bn, bk]`
- `gemm2_weights_scale[expert, hb, ib]`

Each `128 x 128` tile uses one scale pair. Accumulation is block-scaled, not per-channel.

## API

Use:
- `Kernel4Problem` for pointers, shapes, routing metadata, backend, stream
- `Kernel4Workspace` for intermediate/output scratch buffers
- `Kernel6Problem` for standalone GEMM2 + combine
- `Kernel6Workspace` for GEMM2 + combine scratch buffers

Helpers:
- `k4_query_workspace(...)`
- `k4_bind_workspace(...)`
- `k4_launch(...)`
- `k6_query_workspace(...)`
- `k6_bind_workspace(...)`
- `k6_launch(...)`

Backends:
- `Fallback`: slow reference GPU path
- `Tiled`: hand-written tiled GEMM1 path
- `Cutlass`: correctness-first CUTLASS path
- `Auto`: prefers CUTLASS when available, otherwise falls back to `Fallback`

`kernel6` supports `Fallback`, `Cutlass`, and `Auto`.

## Source Layout

- `kernel4.cu`: public API, workspace binding, backend dispatch
- `kernel4_cuda_kernels.cu`: CUDA kernels
- `kernel4_backends.cu`: fallback and tiled launchers
- `kernel4_cutlass.cu`: CUTLASS backend
- `kernel4_cpu_reference.cu`: CPU reference
- `kernel4_internal.cuh`: internal helpers and declarations
- `kernel6.cu`: public Kernel 6 API
- `kernel6_backends.cu`: fallback GEMM2 + combine
- `kernel6_cutlass.cu`: CUTLASS GEMM2 + combine
- `kernel6_cuda_kernels.cu`: shared GEMM2/combine CUDA kernels
- `kernel6_cpu_reference.cu`: CPU reference
- `kernel6_internal.cuh`: internal Kernel 6 helpers

## Build

Fallback:

```bash
make build-fallback
make test-fallback
```

CUTLASS:

```bash
make build-cutlass CUTLASS_DIR=/path/to/cutlass
make test-cutlass CUTLASS_DIR=/path/to/cutlass
```

Useful targets:

```bash
make help
make config
make bench-fallback
make bench-cutlass CUTLASS_DIR=/path/to/cutlass
```

## Tests

`test/test.cu` includes:
- FP8 encode/decode checks
- block-scale math checks
- SwiGLU checks
- routing checks
- end-to-end launch checks
- backend benchmark

## Status

- `kernel4` fallback, tiled, and CUTLASS paths are wired and tested.
- `kernel6` fallback and CUTLASS paths are wired and tested.
- Current CUTLASS paths use small-batch grouped GEMM, but still materialize dense staging buffers.
- On the current RTX 3090 setup, `Cutlass` is the fastest backend when enabled.

## TODO
- Move SwiGLU into a CUTLASS epilogue.
