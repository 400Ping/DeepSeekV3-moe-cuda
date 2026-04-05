#!/bin/bash
# Use paths relative to the uv environment
VENV_PATH=$(pwd)/.venv
NVCC=nvcc
TVM_FFI_PATH=$VENV_PATH/lib/python3.12/site-packages/tvm_ffi

$NVCC -shared -Xcompiler -fPIC \
    -arch=sm_90 \
    -I$TVM_FFI_PATH/include \
    -L$TVM_FFI_PATH/lib \
    -ltvm_ffi \
    -o librouter_ffi.so \
    router_ffi.cu
