#pragma once

#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <optional>

namespace c1_residual_rms {

inline void check_no_overlap(
    const at::Tensor& output,
    const char* output_name,
    const at::Tensor& input,
    const char* input_name
) {
    if (!output.is_alias_of(input)) {
        return;
    }
    TORCH_CHECK(
        at::get_overlap_status(output, input) == at::MemOverlapStatus::No,
        "C1 output ", output_name, " must not overlap ", input_name
    );
}

inline void check_output_overlap_contract(
    const at::Tensor& D,
    const at::Tensor& row_rms_partial,
    const at::Tensor& R,
    const std::optional<at::Tensor>& gamma_opt
) {
    check_no_overlap(D, "D", R, "R");
    check_no_overlap(D, "D", row_rms_partial, "row_rms_partial");
    check_no_overlap(row_rms_partial, "row_rms_partial", R, "R");
    if (gamma_opt.has_value()) {
        check_no_overlap(D, "D", gamma_opt.value(), "gamma");
        check_no_overlap(row_rms_partial, "row_rms_partial", gamma_opt.value(), "gamma");
    }
}

}  // namespace c1_residual_rms
