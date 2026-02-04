#include "rope_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids,
           size_t seqlen, size_t nhead, size_t head_dim, float theta) {
    size_t dim_half = head_dim / 2;

    for (size_t s = 0; s < seqlen; ++s) {
        int64_t pos = pos_ids[s];

        // 计算当前 token 的起始偏移量
        // 输入是 [seqlen, nhead, head_dim] row-major
        // stride_seq = nhead * head_dim
        // stride_head = head_dim
        size_t seq_offset = s * nhead * head_dim;

        for (size_t h = 0; h < nhead; ++h) {
            size_t head_offset = seq_offset + h * head_dim;

            // 指向当前 Head 向量的起始位置
            const T *vec_in = in + head_offset;
            T *vec_out = out + head_offset;

            // 遍历前半段维度
            for (size_t j = 0; j < dim_half; ++j) {
                float a_val = 0.0f, b_val = 0.0f;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    a_val = llaisys::utils::cast<float>(vec_in[j]);
                    b_val = llaisys::utils::cast<float>(vec_in[j + dim_half]);
                } else {
                    a_val = vec_in[j];
                    b_val = vec_in[j + dim_half];
                }

                // 计算角度 phi
                // 使用 double 计算中间变量以减少误差
                double theta_db = static_cast<double>(theta);
                double freq = 1.0 / std::pow(theta_db, (2.0 * j / head_dim));
                double val = static_cast<double>(pos) * freq;

                float cos_val = static_cast<float>(std::cos(val));
                float sin_val = static_cast<float>(std::sin(val));

                // 执行旋转
                float out_a = a_val * cos_val - b_val * sin_val;
                float out_b = b_val * cos_val + a_val * sin_val;

                // 写入结果
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    vec_out[j] = llaisys::utils::cast<T>(out_a);
                    vec_out[j + dim_half] = llaisys::utils::cast<T>(out_b);
                } else {
                    vec_out[j] = out_a;
                    vec_out[j + dim_half] = out_b;
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t dtype, size_t seqlen, size_t nhead, size_t head_dim, float theta) {
    const int64_t *pos_ptr = reinterpret_cast<const int64_t *>(pos_ids);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out),
                     reinterpret_cast<const float *>(in),
                     pos_ptr,
                     seqlen, nhead, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            pos_ptr,
            seqlen, nhead, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            pos_ptr,
            seqlen, nhead, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu