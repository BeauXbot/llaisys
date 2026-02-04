#include "../../../utils.hpp"
#include "swiglu_cpu.hpp"
#include <cmath>

namespace {

template <typename T>
void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; ++i) {
        float g_val, u_val;

        // Load and cast to float
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            g_val = llaisys::utils::cast<float>(gate[i]);
            u_val = llaisys::utils::cast<float>(up[i]);
        } else {
            g_val = gate[i];
            u_val = up[i];
        }

        // silu(x) = x / (1 + exp(-x))
        float silu_val = g_val / (1.0f + std::exp(-g_val));
        float result = u_val * silu_val;

        // Store back
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            out[i] = llaisys::utils::cast<T>(result);
        } else {
            out[i] = result;
        }
    }
}

} // namespace

namespace llaisys::ops::cpu {

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, size_t numel, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        swiglu_kernel(reinterpret_cast<float *>(out),
                      reinterpret_cast<const float *>(gate),
                      reinterpret_cast<const float *>(up),
                      numel);
        break;
    case LLAISYS_DTYPE_BF16:
        swiglu_kernel(reinterpret_cast<llaisys::bf16_t *>(out),
                      reinterpret_cast<const llaisys::bf16_t *>(gate),
                      reinterpret_cast<const llaisys::bf16_t *>(up),
                      numel);
        break;
    case LLAISYS_DTYPE_F16:
        swiglu_kernel(reinterpret_cast<llaisys::fp16_t *>(out),
                      reinterpret_cast<const llaisys::fp16_t *>(gate),
                      reinterpret_cast<const llaisys::fp16_t *>(up),
                      numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu