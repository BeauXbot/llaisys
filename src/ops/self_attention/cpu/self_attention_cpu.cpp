#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

template <typename T>
void self_attention_kernel(T *out, const T *q, const T *k, const T *v, size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv, float scale) {
    
}

namespace llaisys::ops::cpu {

void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t dtype,
                    size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead,
                    size_t d, size_t dv, float scale) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        self_attention_kernel(
            reinterpret_cast<float *>(out), reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v),
            seqlen, total_len, nhead, nkvhead, d, dv, scale);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attention_kernel(
            reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(q),
            reinterpret_cast<const llaisys::bf16_t *>(k), reinterpret_cast<const llaisys::bf16_t *>(v),
            seqlen, total_len, nhead, nkvhead, d, dv, scale);
        break;
    case LLAISYS_DTYPE_F16:
        self_attention_kernel(
            reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(q),
            reinterpret_cast<const llaisys::fp16_t *>(k), reinterpret_cast<const llaisys::fp16_t *>(v),
            seqlen, total_len, nhead, nkvhead, d, dv, scale);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu