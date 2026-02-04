#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "SwiGLU: inputs must be contiguous.");

    // Shape validation
    // out, gate, up: [seqlen, intermediate_size]
    ASSERT(gate->ndim() == 2, "SwiGLU: gate must be 2D.");
    ASSERT(up->ndim() == 2, "SwiGLU: up must be 2D.");
    ASSERT(out->ndim() == 2, "SwiGLU: out must be 2D.");

    size_t rows = gate->shape()[0];
    size_t cols = gate->shape()[1];

    ASSERT(up->shape()[0] == rows && up->shape()[1] == cols, "SwiGLU: shape mismatch (up vs gate).");
    ASSERT(out->shape()[0] == rows && out->shape()[1] == cols, "SwiGLU: shape mismatch (out vs gate).");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::swiglu(
            out->data(),
            gate->data(),
            up->data(),
            rows * cols, // 视为打平的 1D 数组处理
            out->dtype());
        return;
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(
            out->data(),
            gate->data(),
            up->data(),
            rows * cols, // 视为打平的 1D 数组处理
            out->dtype());

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
