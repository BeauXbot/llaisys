#include "embedding_cpu.hpp"
#include "../../../utils.hpp"
#include <cstring> // for std::memcpy

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t num_tokens, size_t hidden_dim, size_t vocab_size) {
    size_t row_size_bytes = hidden_dim * sizeof(T);

    for (size_t i = 0; i < num_tokens; ++i) {
        int64_t idx = index[i];

        if (idx < 0 || (size_t)idx >= vocab_size) {
            throw std::runtime_error("Embedding index out of range");
        }

        const T *src_ptr = weight + idx * hidden_dim;
        T *dst_ptr = out + i * hidden_dim;

        std::memcpy(dst_ptr, src_ptr, row_size_bytes);
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t dtype, size_t num_tokens, size_t hidden_dim, size_t vocab_size) {
    // 处理index
    const int64_t *idx_ptr = reinterpret_cast<const int64_t *>(index);

    // 函数分发
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), idx_ptr, reinterpret_cast<const float *>(weight), num_tokens, hidden_dim, vocab_size);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<bf16_t *>(out), idx_ptr, reinterpret_cast<const llaisys::bf16_t *>(weight), num_tokens, hidden_dim, vocab_size);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<fp16_t *>(out), idx_ptr, reinterpret_cast<const llaisys::fp16_t *>(weight), num_tokens, hidden_dim, vocab_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu