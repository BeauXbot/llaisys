#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    // check
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "RMSNorm: inputs must be contiguous.");

    // 2. 形状检查
    // in: [M, d]  通常是 [Batch, Hidden]
    // weight: [d]
    // out: [M, d]
    ASSERT(in->ndim() >= 1, "RMSNorm: input must be at least 1D.");
    size_t last_dim = in->shape().back();
    // M (不管前几维是多少，拍平成 Rows)
    size_t num_rows = in->numel() / last_dim;

    ASSERT(weight->ndim() == 1, "RMSNorm: weight must be 1D.");
    ASSERT(weight->shape()[0] == last_dim, "RMSNorm: weight dim mismatch.");

    ASSERT(out->numel() == in->numel(), "RMSNorm: output size mismatch.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::rms_norm(
            out->data(),
            in->data(),
            weight->data(),
            out->dtype(),
            num_rows,
            last_dim,
            eps);
        return;
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
