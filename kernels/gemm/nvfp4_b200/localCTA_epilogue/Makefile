NVCC ?= /usr/local/cuda/bin/nvcc
OUT := _C_nv_localcta_gemm$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

NVCCFLAGS := -std=c++17 -O3
NVCCFLAGS += --expt-relaxed-constexpr
NVCCFLAGS += -forward-unknown-to-host-compiler
NVCCFLAGS += -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing
NVCCFLAGS += -DNDEBUG -lineinfo
NVCCFLAGS += -shared -fPIC
NVCCFLAGS += -diag-suppress 3189,3288
NVCCFLAGS += -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=_C_nv_localcta_gemm
NVCCFLAGS += -gencode arch=compute_100a,code=sm_100a

PYTHON_INCLUDES := $(shell python3 -c "import sysconfig; print('-I' + sysconfig.get_path('include'))")
PYBIND_INCLUDES := $(shell python3 -m pybind11 --includes)
PYTORCH_INCLUDES := $(shell python3 -c "from torch.utils.cpp_extension import include_paths; print(' '.join(['-I' + p for p in include_paths()]))")
PYTORCH_LIBRARY_PATHS := $(shell python3 -c "from torch.utils.cpp_extension import library_paths; print(' '.join(library_paths()))")
PYTORCH_LIBDIR := $(foreach p,$(PYTORCH_LIBRARY_PATHS),-L$(p))
PYTORCH_RPATH := $(foreach p,$(PYTORCH_LIBRARY_PATHS),-Xlinker -rpath -Xlinker $(p))

NVCCFLAGS += $(PYTHON_INCLUDES) $(PYBIND_INCLUDES) $(PYTORCH_INCLUDES) $(PYTORCH_LIBDIR) $(PYTORCH_RPATH)
NVCCFLAGS += -ltorch_python -ltorch_cuda -ltorch_cpu -ltorch -lc10_cuda -lc10

KERNEL_HEADERS := ../../../../../TK_quantisation/nvfp4_CTA_local_v1/localcta_reconstruct.cuh

all: $(OUT)

$(OUT): nvfp4_localcta_gemm.cu $(KERNEL_HEADERS)
	$(NVCC) nvfp4_localcta_gemm.cu $(NVCCFLAGS) -o $(OUT)

clean:
	rm -f $(OUT)

.PHONY: all clean
