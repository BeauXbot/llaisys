#include "rms_norm_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t M, size_t d, float eps) {
    for (size_t i = 0; i < M; ++i) {
        const T *row_in = in + i * d;
        T *row_out = out + i * d;

        float sqr_sum = 0.0f;
        for (size_t j = 0; j < d; ++j) {
            float val = 0.0f;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                val = llaisys::utils::cast<float>(row_in[j]);
            } else {
                val = row_in[j];
            }
            sqr_sum += val * val;
        }

        // rms = sqrt(mean(x^2) + eps)
        float rms = std::sqrt(sqr_sum / d + eps);

        // 预计算缩放因子的倒数，把除法变乘法
        float inv_rms = 1.0f / rms;

        for (size_t j = 0; j < d; ++j) {
            float val = 0.0f, w = 0.0f;

            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                val = llaisys::utils::cast<float>(row_in[j]);
                w = llaisys::utils::cast<float>(weight[j]);
            } else {
                val = row_in[j];
                w = weight[j];
            }

            float res = (val * inv_rms) * w;

            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                row_out[j] = llaisys::utils::cast<T>(res);
            } else {
                row_out[j] = res;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t dtype, size_t M, size_t d, float eps) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            M, d, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            M, d, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            M, d, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu