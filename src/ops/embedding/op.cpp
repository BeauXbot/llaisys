#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), "Embedding: inputs must be contiguous.");

    // index
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index must be Int64.");
    // out 和 weight 必须类型一致
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());

    // 检查 weight: [vocab_size, hidden_dim]
    ASSERT(weight->ndim() == 2, "Embedding: weight must be 2D.");
    size_t vocab_size = weight->shape()[0];
    size_t hidden_dim = weight->shape()[1];

    size_t num_tokens = index->numel();

    // out: [num_tokens, hidden_dim]
    // ASSERT(out->numel == num_tokens * hidden_dim, "Embedding: output size mismatch.");
    ASSERT(out->ndim() == 2 && out->shape()[0] == num_tokens && out->shape()[1] == hidden_dim, "Embedding: output size mismatch.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::embedding(
            out->data(),
            index->data(),
            weight->data(),
            out->dtype(),
            num_tokens,
            hidden_dim,
            vocab_size);
        return;
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {

    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(
            out->data(),
            index->data(),
            weight->data(),
            out->dtype(),
            num_tokens,
            hidden_dim,
            vocab_size);
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
