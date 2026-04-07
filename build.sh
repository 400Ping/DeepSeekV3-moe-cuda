#!/bin/bash
# Use paths relative to the uv environment
VENV_PATH=$(pwd)/.venv
NVCC=nvcc
TVM_FFI_PATH=$VENV_PATH/lib/python3.12/site-packages/tvm_ffi

echo "Building MoE FFI library..."
$NVCC -shared -Xcompiler -fPIC \
    -arch=sm_90 \
    -I$TVM_FFI_PATH/include \
    -L$TVM_FFI_PATH/lib \
    -ltvm_ffi \
    -o librouter_ffi.so \
    kernels/moe_ffi.cu

echo "Building MoE Scan standalone test..."
$NVCC -arch=sm_90 -o tests/test_scan tests/test_moe_scan_cuda.cu

echo "Building MoE Expert MLP tests..."
make -C kernels/moe_expert_mlp CUDA_ARCH=sm_90 build-fallback

