GPU := B200
SRC := bf16_b200_gemm.cu
OUT := bf16_b200_gemm.out
CMD := ./bf16_b200_gemm.out
CONFIG := standalone

ifeq ($(CONFIG),pytorch)
NVCCFLAGS += -DTORCH_COMPILE
endif

# PyTorch extension smoke:
#   make clean && make CONFIG=pytorch \
#     OUT=$$(python3 - <<'PY'
# import sysconfig
# print('_C' + sysconfig.get_config_var('EXT_SUFFIX'))
# PY
#     ) \
#     CMD='python3 test_gemm.py 512 256 64'
include ../../common.mk
