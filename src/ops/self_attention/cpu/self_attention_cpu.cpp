#include "../../../utils.hpp"
#include "self_attention_cpu.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

template <typename T>
void self_attention_kernel(T *out, const T *q, const T *k, const T *v,
                           size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead,
                           size_t d, size_t dv, float scale) {
    size_t group_size = nhead / nkvhead;

    // 为了避免频繁分配，我们在外层分配一个 scores 缓冲区
    // 但由于是多层循环，且 total_len 可能很大，这里简单起见每次声明
    // 实际优化应当重用内存

    // row-major strides
    // q: [seqlen, nhead, d] -> stride_seq = nhead*d, stride_head = d
    // k: [total_len, nkvhead, d]
    // v: [total_len, nkvhead, dv]
    // out: [seqlen, nhead, dv]

    for (size_t s = 0; s < seqlen; ++s) {
        for (size_t h = 0; h < nhead; ++h) {
            size_t kv_h = h / group_size;

            // 定位当前 token、当前 head 的 Query 向量
            const T *q_vec = q + (s * nhead + h) * d;

            // 1. Q * K^T calculation
            std::vector<float> scores(total_len);
            float max_score = -std::numeric_limits<float>::infinity();

            for (size_t t = 0; t < total_len; ++t) {
                // Causal Masking:
                // 我们需要支持 Q 是 K 的后缀子集的情况 (qlen <= total_len)
                // PyTorch 测试中使用的是 torch.ones(L, S).tril(diagonal=S-L)
                // 这意味着有效位置满足: t <= s + (total_len - seqlen)
                // 相反，被 Mask 的位置是: t > s + (total_len - seqlen)

                // 注意: 仅当 seqlen > 1 (Prefill/Prompt阶段) 时通常才需要显式mask，
                // 但这个通用公式也适用于 seqlen=1 (Decode) 的情况 (此时 s=0, Mask 条件为 t > total_len-1，即永不mask)
                bool masked = false;
                if (total_len >= seqlen) {
                    if (t > s + (total_len - seqlen)) {
                        masked = true;
                    }
                } else {
                    // 异常情况保护(虽不常见): q比k长?
                    // 遵循对角线逻辑: t + (seqlen - total_len) > s
                    if (t + (seqlen - total_len) > s) {
                        masked = true;
                    }
                }

                if (masked) {
                    scores[t] = -std::numeric_limits<float>::infinity();
                    continue;
                }

                // 定位 Key 向量: k[t, kv_h, :]
                const T *k_vec = k + (t * nkvhead + kv_h) * d;

                float dot = 0.0f;
                for (size_t i = 0; i < d; ++i) {
                    float q_val, k_val;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        q_val = llaisys::utils::cast<float>(q_vec[i]);
                        k_val = llaisys::utils::cast<float>(k_vec[i]);
                    } else {
                        q_val = static_cast<float>(q_vec[i]);
                        k_val = static_cast<float>(k_vec[i]);
                    }
                    dot += q_val * k_val;
                }
                scores[t] = dot * scale;
                if (scores[t] > max_score) {
                    max_score = scores[t];
                }
            }

            // 2. Softmax
            float sum_exp = 0.0f;
            for (size_t t = 0; t < total_len; ++t) {
                if (scores[t] <= -1e30f) { // 近似 infinity check
                    scores[t] = 0.0f;
                } else {
                    float val = std::exp(scores[t] - max_score);
                    scores[t] = val;
                    sum_exp += val;
                }
            }
            // 归一化
            // (如果没有有效位置 sum_exp 可能为 0，防止除零)
            float inv_sum = (sum_exp > 1e-10f) ? (1.0f / sum_exp) : 0.0f;

            // 3. Score * V -> Output
            T *out_vec = out + (s * nhead + h) * dv;
            for (size_t i = 0; i < dv; ++i) {
                float acc = 0.0f;
                for (size_t t = 0; t < total_len; ++t) {
                    // 如果 score 为 0，跳过计算
                    if (scores[t] == 0.0f) {
                        continue;
                    }

                    const T *v_vec = v + (t * nkvhead + kv_h) * dv;
                    float v_val;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        v_val = llaisys::utils::cast<float>(v_vec[i]);
                    } else {
                        v_val = static_cast<float>(v_vec[i]);
                    }
                    acc += scores[t] * inv_sum * v_val;
                }

                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out_vec[i] = llaisys::utils::cast<T>(acc);
                } else {
                    out_vec[i] = static_cast<T>(acc);
                }
            }
        }
    }
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