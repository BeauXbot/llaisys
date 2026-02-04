#include "linear_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             size_t M, size_t K, size_t N) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            const T *in_row = in + i * K;
            const T *w_row = weight + j * K;

            for (size_t k = 0; k < K; ++k) {
                float v_in = 0.0f, v_w = 0.0f;

                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    v_in = llaisys::utils::cast<float>(in_row[k]);
                    v_w = llaisys::utils::cast<float>(w_row[k]);
                }else{
                    v_in = in_row[k];
                    v_w = w_row[k];
                }

                sum += v_in * v_w;
            }

            if(bias){
                float v_b = 0.0f;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>){
                    v_b = llaisys::utils::cast<float>(bias[j]);
                }else{
                    v_b = bias[j];
                }
                sum += v_b;
            }

            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>){
                out[i * N + j] = llaisys::utils::cast<T>(sum);
            }else{
                out[i * N + j] = sum;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t dtype, size_t M, size_t K, size_t N) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return linear_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            reinterpret_cast<const float *>(bias),
            M, K, N);
    case LLAISYS_DTYPE_BF16:
        return linear_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            reinterpret_cast<const llaisys::bf16_t *>(bias),
            M, K, N);
    case LLAISYS_DTYPE_F16:
        return linear_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            reinterpret_cast<const llaisys::fp16_t *>(bias),
            M, K, N);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu