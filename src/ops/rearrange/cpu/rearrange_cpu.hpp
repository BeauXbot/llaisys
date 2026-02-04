#pragma once
#include "llaisys.h"
#include <cstddef>
#include <cstdint> // for ptrdiff_t
#include <vector>

namespace llaisys::ops::cpu {

void rearrange(std::byte *out, const std::byte *in,
               const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &out_strides, // size_t -> ptrdiff_t
               const std::vector<ptrdiff_t> &in_strides,  // size_t -> ptrdiff_t
               size_t elem_size);

} // namespace llaisys::ops::cpu