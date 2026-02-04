#include "op.hpp"

// 包含核心库，用于获取 context (管理设备上下文)
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp" // 包含断言宏 CHECK_XXX

#include "cpu/argmax_cpu.hpp" // 引入 CPU 实现的头文件

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // 1.基本检查
    // 如果有 max_val，检查设备一致性
    if (max_val) {
        CHECK_SAME_DEVICE(max_idx, max_val, vals);
        CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
        ASSERT(max_val->isContiguous(), "Argmax: max_val must be contiguous");
    } else {
        CHECK_SAME_DEVICE(max_idx, vals);
    }

    ASSERT(vals->isContiguous() && max_idx->isContiguous(), "Argmax: inputs must be contiguous.");
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64 || max_idx->dtype() == LLAISYS_DTYPE_I32, "Argmax: output index must be I64 or I32.");

    size_t numel = vals->numel();

    ASSERT(max_idx->numel() >= 1, "Argmax: output index tensor too small");
    if (max_val) {
        ASSERT(max_val->numel() >= 1, "Argmax: output value tensor too small");
    }

    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        // 调用CPU的实现
        cpu::argmax(
            max_idx->data(),
            max_val ? max_val->data() : nullptr,
            vals->data(),
            vals->dtype(),
            numel);
        return;
    }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());
    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(
            max_idx->data(),
            max_val ? max_val->data() : nullptr,
            vals->data(),
            vals->dtype(),
            numel);
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
