GPU := B200

# Standalone mode
# SRC := mxfp4_gb200_gemm.cu
# OUT := mxfp4_gb200_gemm.out
# CMD := ./mxfp4_gb200_gemm.out
# CONFIG := standalone

# PyTorch extension mode
SRC := mxfp4_gb200_gemm.cu
OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := python test_gemm.py # or test_quantize.py
NVCCFLAGS := -DTORCH_COMPILE
CONFIG := pytorch

include ../../common.mk
