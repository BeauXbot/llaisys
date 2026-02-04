#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t dtype, size_t num_tokens, size_t hidden_size, size_t vocab_size);
}