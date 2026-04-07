#!/bin/bash
# Unified test script for MoE CUDA kernels

# 1. Python tests (Routing & Dispatch)
echo "Running Python tests (uv run pytest)..."
uv run python -m pytest tests/test_moe_routing.py tests/test_moe_dispatch.py

# 2. CUDA Scan standalone test
if [ -f tests/test_scan ]; then
    echo "Running CUDA Scan test..."
    ./tests/test_scan
else
    echo "Error: tests/test_scan not found. Please run ./build.sh first."
fi

# 3. CUDA Expert MLP test
if [ -f kernels/moe_expert_mlp/.build/kernel4_test_fallback ]; then
    echo "Running CUDA Expert MLP test..."
    ./kernels/moe_expert_mlp/.build/kernel4_test_fallback
else
    echo "Error: kernels/moe_expert_mlp/.build/kernel4_test_fallback not found. Please run ./build.sh first."
fi
