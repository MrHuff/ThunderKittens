#pragma once

#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>

namespace c3_row_scale {

inline void check_output_no_overlap(
    const at::Tensor& D,
    const c10::StorageImpl* d_storage,
    const at::Tensor& input,
    const char* input_name
) {
    if (d_storage != input.storage().unsafeGetStorageImpl()) {
        return;
    }
    TORCH_CHECK(
        at::get_overlap_status(D, input) == at::MemOverlapStatus::No,
        "C3 output D must not overlap ", input_name
    );
}

inline const c10::StorageImpl* check_common(
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& row_scale_coeff,
    const at::Tensor& D
) {
    TORCH_CHECK(
        row_scale_coeff.scalar_type() == at::kFloat,
        "C3 row_scale_coeff must be float32"
    );
    TORCH_CHECK(row_scale_coeff.is_contiguous(), "C3 row_scale_coeff must be contiguous");
    TORCH_CHECK(
        row_scale_coeff.dim() == 1 && row_scale_coeff.size(0) == A.size(0),
        "C3 row_scale_coeff must have shape [M]"
    );
    TORCH_CHECK(D.is_contiguous(), "C3 output D must be contiguous");
    TORCH_CHECK(D.dim() == 2, "C3 output D must be 2D");
    TORCH_CHECK(D.scalar_type() == at::kBFloat16, "C3 output D must be bf16");
    TORCH_CHECK(
        D.size(0) == A.size(0) && D.size(1) == B.size(0),
        "C3 output D shape mismatch"
    );
    const auto* d_storage = D.storage().unsafeGetStorageImpl();
    check_output_no_overlap(D, d_storage, A, "A");
    check_output_no_overlap(D, d_storage, B, "B");
    check_output_no_overlap(D, d_storage, row_scale_coeff, "row_scale_coeff");
    return d_storage;
}

inline void check_nvfp4_contract(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg,
    const at::Tensor& row_scale_coeff,
    const at::Tensor& D
) {
    const auto* d_storage = check_common(A, B, row_scale_coeff, D);
    check_output_no_overlap(D, d_storage, A_sc, "A_sc");
    check_output_no_overlap(D, d_storage, A_sg, "A_sg");
    check_output_no_overlap(D, d_storage, B_sc, "B_sc");
    check_output_no_overlap(D, d_storage, B_sg, "B_sg");
}

inline void check_mxfp4_contract(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& row_scale_coeff,
    const at::Tensor& D
) {
    const auto* d_storage = check_common(A, B, row_scale_coeff, D);
    check_output_no_overlap(D, d_storage, A_sc, "A_sc");
    check_output_no_overlap(D, d_storage, B_sc, "B_sc");
}

}  // namespace c3_row_scale
