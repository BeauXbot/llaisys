#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rearrange_cpu.hpp"

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    // Rearrange 允许步长不同，但形状必须完全一致
    ASSERT(out->ndim() == in->ndim(), "Rearrange: input and output must have same ndim.");
    for (size_t i = 0; i < out->ndim(); ++i) {
        ASSERT(out->shape()[i] == in->shape()[i], "Rearrange: shape mismatch.");
    }

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::rearrange(
            out->data(),
            in->data(),
            out->shape(),
            out->strides(),
            in->strides(),
            out->elementSize() // 修改这里: itemsize() -> elementSize()
        );
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
