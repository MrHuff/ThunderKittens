// ================================================================
// MXFP4 Fused Cross-Entropy — Compilation unit + PyBind11
// ================================================================
#include "mxfp4_cce.cuh"

#ifndef TORCH_COMPILE

// Standalone mode not implemented — CCE is PyTorch-only
int main() { return 0; }

#else

#include "pyutils/torchutils.cuh"

void mxfp4_cce_entrypoint(
    const at::Tensor &A,       // [M_pad, K/2] uint8
    const at::Tensor &A_sc,    // [M_pad/128, K/128, 32, 16] uint8 E8M0
    const at::Tensor &B,       // [N_pad, K/2] uint8
    const at::Tensor &B_sc,    // [N_pad/128, K/128, 32, 16] uint8 E8M0
    at::Tensor &lse,           // [M] float32 — initialized to -inf
    at::Tensor &neg_logit,     // [M] float32 — initialized to 0
    const at::Tensor &targets, // [M] int64
    int M,                     // actual M (unpadded)
    int N                      // actual N (unpadded)
) {
    using C = mxfp4_cce::config<256, 5, 8, 4, 2, false>;
    using G = mxfp4_cce::globals<C>;

    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .lse = lse.data_ptr<float>(),
        .neg_correct_logit = neg_logit.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .M = M,
        .N = N
    };

    // Use kittens::py::launch_kernel which handles cluster launch, PDL, and shared memory
    kittens::py::launch_kernel<C, G, mxfp4_cce::kernel<C>>(g);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mxfp4_cce", &mxfp4_cce_entrypoint,
          "MXFP4 Fused Cross-Entropy (no materialized logits)",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("lse"), pybind11::arg("neg_logit"),
          pybind11::arg("targets"),
          pybind11::arg("M"), pybind11::arg("N"));
}

#endif
