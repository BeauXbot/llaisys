#include "argmax_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <limits> // for std::numeric_limits

template <typename T>
void argmax_(int64_t *max_out, T *val_out, const T *vals, size_t numel) {
    if (numel == 0) {
        return;
    }

    float max_v = -std::numeric_limits<float>::infinity();
    int64_t max_i = 0;

    // 如果是 F16/BF16，先转 float 再比较
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        max_v = llaisys::utils::cast<float>(vals[0]);
    } else {
        max_v = vals[0];
    }

    for (size_t i = 1; i < numel; ++i) {
        float curr = 0;
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            curr = llaisys::utils::cast<float>(vals[i]);
        } else {
            curr = vals[i];
        }

        if (curr > max_v) {
            max_v = curr;
            max_i = i;
        }
    }

    // 写入结果
    max_out[0] = max_i;
    if (val_out) {
        // 如果需要 max_val，需要把 float 类型的 max_v 转回 T
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            val_out[0] = llaisys::utils::cast<T>(max_v);
        } else {
            val_out[0] = max_v;
        }
    }
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<int64_t *>(max_idx),
                       reinterpret_cast<float *>(max_val),
                       reinterpret_cast<const float *>(vals),
                       numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx),
                       reinterpret_cast<llaisys::bf16_t *>(max_val),
                       reinterpret_cast<const llaisys::bf16_t *>(vals),
                       numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx),
                       reinterpret_cast<llaisys::fp16_t *>(max_val),
                       reinterpret_cast<const llaisys::fp16_t *>(vals),
                       numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu