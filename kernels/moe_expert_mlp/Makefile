NVCC ?= nvcc
CUDA_ARCH ?= sm_86
OUT_DIR ?= .build
TARGET_BASE ?= kernel4_test
ENABLE_CUTLASS ?= 0

DEFAULT_CUTLASS_DIR := $(HOME)/.local/src/cutlass
ifeq ($(wildcard $(DEFAULT_CUTLASS_DIR)/include/cutlass/cutlass.h),)
DEFAULT_CUTLASS_DIR :=
endif
CUTLASS_DIR ?= $(DEFAULT_CUTLASS_DIR)

SRC_DIR := .
TEST_DIR := test
TEST_DIR := ../../tests

SOURCES := \
	$(SRC_DIR)/kernel4.cu \
	$(SRC_DIR)/kernel4_cuda_kernels.cu \
	$(SRC_DIR)/kernel4_backends.cu \
	$(SRC_DIR)/kernel4_cpu_reference.cu \
	$(SRC_DIR)/kernel4_cutlass.cu \
	$(SRC_DIR)/kernel6.cu \
	$(SRC_DIR)/kernel6_cuda_kernels.cu \
	$(SRC_DIR)/kernel6_backends.cu \
	$(SRC_DIR)/kernel6_cpu_reference.cu \
	$(SRC_DIR)/kernel6_cutlass.cu \
	$(TEST_DIR)/test_moe_expert_mlp_cuda.cu
HEADERS := \
	$(SRC_DIR)/kernel4.cuh \
	$(SRC_DIR)/kernel4_internal.cuh \
	$(SRC_DIR)/kernel6.cuh \
	$(SRC_DIR)/kernel6_internal.cuh

COMMON_INCLUDES := -I$(SRC_DIR) -I$(TEST_DIR)
COMMON_NVCCFLAGS := -std=c++17 -O2 -arch=$(CUDA_ARCH) $(COMMON_INCLUDES)
FALLBACK_NVCCFLAGS := $(COMMON_NVCCFLAGS)
CUTLASS_NVCCFLAGS := $(COMMON_NVCCFLAGS) -I$(CUTLASS_DIR)/include -DK4_ENABLE_CUTLASS=1 --expt-relaxed-constexpr

FALLBACK_TARGET := $(OUT_DIR)/$(TARGET_BASE)_fallback
CUTLASS_TARGET := $(OUT_DIR)/$(TARGET_BASE)_cutlass

AUTO_TARGET := $(FALLBACK_TARGET)
ifeq ($(ENABLE_CUTLASS),1)
AUTO_TARGET := $(CUTLASS_TARGET)
endif

.PHONY: help config all build build-fallback build-cutlass \
	run run-fallback run-cutlass \
	test test-fallback test-cutlass bench bench-fallback bench-cutlass \
	rebuild rebuild-fallback rebuild-cutlass \
	clean require-cutlass

help:
	@echo "Targets:"
	@echo "  build             Build default binary (ENABLE_CUTLASS=$(ENABLE_CUTLASS))"
	@echo "  build-fallback    Build fallback/reference binary"
	@echo "  build-cutlass     Build CUTLASS-enabled binary"
	@echo "  run               Run default binary"
	@echo "  run-fallback      Run fallback/reference binary"
	@echo "  run-cutlass       Run CUTLASS-enabled binary"
	@echo "  test              Alias of run"
	@echo "  test-fallback     Run fallback/reference tests"
	@echo "  test-cutlass      Run CUTLASS-enabled tests"
	@echo "  bench             Alias of run (binary includes T7 benchmark)"
	@echo "  config            Print resolved build configuration"
	@echo "  clean             Remove build artifacts"

config:
	@echo "NVCC=$(NVCC)"
	@echo "CUDA_ARCH=$(CUDA_ARCH)"
	@echo "OUT_DIR=$(OUT_DIR)"
	@echo "ENABLE_CUTLASS=$(ENABLE_CUTLASS)"
	@echo "CUTLASS_DIR=$(CUTLASS_DIR)"
	@echo "AUTO_TARGET=$(AUTO_TARGET)"

all: build

build: $(AUTO_TARGET)

build-fallback: $(FALLBACK_TARGET)

build-cutlass: $(CUTLASS_TARGET)

run: build
	./$(AUTO_TARGET)

run-fallback: build-fallback
	./$(FALLBACK_TARGET)

run-cutlass: build-cutlass
	./$(CUTLASS_TARGET)

test: run

test-fallback: run-fallback

test-cutlass: run-cutlass

rebuild: clean build

rebuild-fallback: clean build-fallback

rebuild-cutlass: clean build-cutlass

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

require-cutlass:
	@test -n "$(CUTLASS_DIR)" && test -f "$(CUTLASS_DIR)/include/cutlass/cutlass.h" || \
		{ echo "CUTLASS_DIR must point to a CUTLASS checkout with include/cutlass/cutlass.h"; exit 1; }

$(FALLBACK_TARGET): $(SOURCES) $(HEADERS) | $(OUT_DIR)
	$(NVCC) $(FALLBACK_NVCCFLAGS) $(SOURCES) -o $@

$(CUTLASS_TARGET): $(SOURCES) $(HEADERS) | $(OUT_DIR) require-cutlass
	$(NVCC) $(CUTLASS_NVCCFLAGS) $(SOURCES) -o $@

clean:
	rm -f $(FALLBACK_TARGET) $(CUTLASS_TARGET)
	find $(OUT_DIR) -maxdepth 1 -type f ! -name '.nfs*' -delete 2>/dev/null || true
	rmdir $(OUT_DIR) 2>/dev/null || true
