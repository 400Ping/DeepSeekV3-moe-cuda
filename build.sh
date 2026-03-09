#!/bin/bash
# RTX 4060 has compute capability 8.9
nvcc -shared -Xcompiler -fPIC \
    -arch=sm_89 \
    -I/home/weimin.chen/.local/lib/python3.10/site-packages/tvm_ffi/include \
    -L/home/weimin.chen/.local/lib/python3.10/site-packages/tvm_ffi/lib \
    -ltvm_ffi \
    -o librouter_ffi.so \
    router_ffi.cu
