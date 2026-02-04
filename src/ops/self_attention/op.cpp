#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    // check
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "Ops: self_attention inputs must be contiguous");

    // Shapes Validation
    // q: [seqlen, nhead, d]
    ASSERT(q->ndim() == 3, "q must be 3D [seqlen, nhead, d]");
    size_t seqlen = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t d = q->shape()[2];

    // k: [total_len, nkvhead, d]
    ASSERT(k->ndim() == 3, "k must be 3D [total_len, nkvhead, d]");
    size_t total_len = k->shape()[0];
    size_t nkvhead = k->shape()[1];
    ASSERT(k->shape()[2] == d, "k head_dim must match q");

    // v: [total_len, nkvhead, dv]
    ASSERT(v->ndim() == 3, "v must be 3D [total_len, nkvhead, dv]");
    ASSERT(v->shape()[0] == total_len, "v total_len must match k");
    ASSERT(v->shape()[1] == nkvhead, "v nkvhead must match k");
    size_t dv = v->shape()[2];

    // attn_val: [seqlen, nhead, dv]
    ASSERT(attn_val->ndim() == 3, "attn_val must be 3D [seqlen, nhead, dv]");
    ASSERT(attn_val->shape()[0] == seqlen, "attn_val seqlen mismatch");
    ASSERT(attn_val->shape()[1] == nhead, "attn_val nhead mismatch");
    ASSERT(attn_val->shape()[2] == dv, "attn_val head_dim_v mismatch");

    // GQA Requirement
    ASSERT(nhead % nkvhead == 0, "nhead must be divisible by nkvhead (GQA)");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::self_attention(
            attn_val->data(),
            q->data(),
            k->data(),
            v->data(),
            attn_val->dtype(),
            seqlen, total_len, nhead, nkvhead, d, dv, scale);
        return;
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(
            attn_val->data(),
            q->data(),
            k->data(),
            v->data(),
            attn_val->dtype(),
            seqlen, total_len, nhead, nkvhead, d, dv, scale);
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
