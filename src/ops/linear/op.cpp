#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // check
    if (bias) {
        CHECK_SAME_DEVICE(out, in, weight, bias);
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype(), bias->dtype());
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");
    } else {
        CHECK_SAME_DEVICE(out, in, weight);
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    }
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "Linear: main tensors must be contiguous.");

    ASSERT(in->ndim() == 2, "Linear: input must be 2D.");
    ASSERT(weight->ndim() == 2, "Linear: weight must be 2D.");
    ASSERT(out->ndim() == 2, "Linear: output must be 2D.");

    size_t M = in->shape()[0], K = in->shape()[1];
    size_t N = weight->shape()[0], K_w = weight->shape()[1];
    ASSERT(K == K_w, "Linear: input feature dim mismatch with weight.");
    ASSERT(out->shape()[0] == M && out->shape()[1] == N, "Linear: output shape mismatch.");

    if (bias) {
        ASSERT(bias->ndim() == 1, "Linear: bias must be 1D.");
        ASSERT(bias->shape()[0] == N || bias->shape()[0] == 1, "Linear: bias dim mismatch.");
    }

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::linear(
            out->data(),
            in->data(),
            weight->data(),
            bias ? bias->data() : nullptr,
            out->dtype(),
            M, K, N);

        return;
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(
            out->data(),
            in->data(),
            weight->data(),
            bias ? bias->data() : nullptr,
            out->dtype(),
            M, K, N);
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
