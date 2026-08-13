#pragma once

// Early dependent launch does not improve the production MXFP4 shapes and can
// strand a clustered CTA while its predecessor still owns tensor memory.
#ifndef MXFP4_GEMM_DEFAULT_USE_PDL
#define MXFP4_GEMM_DEFAULT_USE_PDL 0
#endif

namespace mxfp4_launch {

inline constexpr bool default_use_pdl = MXFP4_GEMM_DEFAULT_USE_PDL != 0;

} // namespace mxfp4_launch
