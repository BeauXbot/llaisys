#include "rearrange_cpu.hpp"
#include <cstring>
#include <vector>

namespace {

// 递归函数
void rearrange_recursive(std::byte *out_base, const std::byte *in_base,
                         size_t dim, size_t ndim,
                         ptrdiff_t current_out_offset, ptrdiff_t current_in_offset, // change to ptrdiff_t
                         const std::vector<size_t> &shape,
                         const std::vector<ptrdiff_t> &out_strides, // change to ptrdiff_t
                         const std::vector<ptrdiff_t> &in_strides,  // change to ptrdiff_t
                         size_t elem_size) {

    // Base Case
    if (dim == ndim) {
        // Offset 是以字节为单位的
        std::memcpy(out_base + current_out_offset, in_base + current_in_offset, elem_size);
        return;
    }

    // Recursive Step
    size_t dim_size = shape[dim];
    ptrdiff_t out_stride = out_strides[dim] * static_cast<ptrdiff_t>(elem_size);
    ptrdiff_t in_stride = in_strides[dim] * static_cast<ptrdiff_t>(elem_size);

    for (size_t i = 0; i < dim_size; ++i) {
        rearrange_recursive(
            out_base, in_base,
            dim + 1, ndim,
            current_out_offset + i * out_stride,
            current_in_offset + i * in_stride,
            shape, out_strides, in_strides, elem_size);
    }
}

} // namespace

namespace llaisys::ops::cpu {

void rearrange(std::byte *out, const std::byte *in,
               const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &out_strides,
               const std::vector<ptrdiff_t> &in_strides,
               size_t elem_size) {
    size_t ndim = shape.size();

    size_t numel = 1;
    for (auto s : shape) {
        numel *= s;
    }
    if (numel == 0) {
        return;
    }

    rearrange_recursive(out, in, 0, ndim, 0, 0, shape, out_strides, in_strides, elem_size);
}

} // namespace llaisys::ops::cpu