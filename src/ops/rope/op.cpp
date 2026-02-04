#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids must be I64.");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "RoPE: inputs must be contiguous.");

    // in: [seqlen, nhead, d] ==> ndim=3
    ASSERT(in->ndim() == 3, "RoPE: input must be 3D [seqlen, nhead, head_dim].");
    size_t seqlen = in->shape()[0];
    size_t nhead = in->shape()[1];
    size_t head_dim = in->shape()[2];

    // pos_ids: [seqlen]
    ASSERT(pos_ids->ndim() == 1, "RoPE: pos_ids must be 1D.");
    ASSERT(pos_ids->shape()[0] == seqlen, "RoPE: pos_ids vs input seqlen mismatch.");

    // out: 应该和 in 一样
    ASSERT(out->numel() == in->numel(), "RoPE: output size mismatch.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::rope(
            out->data(),
            in->data(),
            pos_ids->data(),
            out->dtype(),
            seqlen,
            nhead,
            head_dim,
            theta);
        return;
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(
            out->data(),
            in->data(),
            pos_ids->data(),
            out->dtype(),
            seqlen,
            nhead,
            head_dim,
            theta);
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
