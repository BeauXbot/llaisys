#include "llaisys/models/qwen2.h"
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../tensor/tensor.hpp"
#include "../llaisys_tensor.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

using namespace llaisys;

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;

    std::vector<tensor_t> k_cache;
    std::vector<tensor_t> v_cache;

    int64_t current_pos;

    llaisysDeviceType_t main_device;
    int main_device_id;
};

llaisysTensor_t wrap(tensor_t t) {
    if (!t) {
        return nullptr;
    }
    return new LlaisysTensor{t};
}

tensor_t unwrap(llaisysTensor_t t) {
    if (!t) {
        return nullptr;
    }
    return t->tensor;
}

tensor_t create_tensor(const std::vector<size_t> &shape, llaisysDataType_t dtype, llaisysDeviceType_t device, int device_id) {
    auto t = Tensor::create(shape, dtype, device, device_id);
    std::memset(t->data(), 0, t->numel() * t->elementSize());
    return t;
}

extern "C" {

struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    auto model = new LlaisysQwen2Model();
    model->meta = *meta;
    model->main_device = device;
    model->main_device_id = (ndevice > 0 && device_ids) ? device_ids[0] : 0;
    model->current_pos = 0;

    size_t hidden = meta->hs;
    size_t interal = meta->di;
    size_t head_dim = meta->dh;
    size_t q_dim = meta->nh * head_dim;
    size_t kv_dim = meta->nkvh * head_dim;

    model->weights.in_embed = wrap(create_tensor({meta->voc, hidden}, meta->dtype, device, model->main_device_id));
    model->weights.out_embed = wrap(create_tensor({meta->voc, hidden}, meta->dtype, device, model->main_device_id));
    model->weights.out_norm_w = wrap(create_tensor({hidden}, meta->dtype, device, model->main_device_id));

    model->weights.attn_norm_w = new llaisysTensor_t[meta->nlayer];
    model->weights.attn_q_w = new llaisysTensor_t[meta->nlayer];
    model->weights.attn_q_b = new llaisysTensor_t[meta->nlayer];
    model->weights.attn_k_w = new llaisysTensor_t[meta->nlayer];
    model->weights.attn_k_b = new llaisysTensor_t[meta->nlayer];
    model->weights.attn_v_w = new llaisysTensor_t[meta->nlayer];
    model->weights.attn_v_b = new llaisysTensor_t[meta->nlayer];
    model->weights.attn_o_w = new llaisysTensor_t[meta->nlayer];
    model->weights.mlp_norm_w = new llaisysTensor_t[meta->nlayer];
    model->weights.mlp_gate_w = new llaisysTensor_t[meta->nlayer];
    model->weights.mlp_up_w = new llaisysTensor_t[meta->nlayer];
    model->weights.mlp_down_w = new llaisysTensor_t[meta->nlayer];

    for (size_t i = 0; i < meta->nlayer; ++i) {
        model->weights.attn_norm_w[i] = wrap(create_tensor({hidden}, meta->dtype, device, model->main_device_id));
        model->weights.attn_q_w[i] = wrap(create_tensor({q_dim, hidden}, meta->dtype, device, model->main_device_id));
        model->weights.attn_q_b[i] = wrap(create_tensor({q_dim}, meta->dtype, device, model->main_device_id));
        model->weights.attn_k_w[i] = wrap(create_tensor({kv_dim, hidden}, meta->dtype, device, model->main_device_id));
        model->weights.attn_k_b[i] = wrap(create_tensor({kv_dim}, meta->dtype, device, model->main_device_id));
        model->weights.attn_v_w[i] = wrap(create_tensor({kv_dim, hidden}, meta->dtype, device, model->main_device_id));
        model->weights.attn_v_b[i] = wrap(create_tensor({kv_dim}, meta->dtype, device, model->main_device_id));
        model->weights.attn_o_w[i] = wrap(create_tensor({hidden, q_dim}, meta->dtype, device, model->main_device_id));
        model->weights.mlp_norm_w[i] = wrap(create_tensor({hidden}, meta->dtype, device, model->main_device_id));
        model->weights.mlp_gate_w[i] = wrap(create_tensor({interal, hidden}, meta->dtype, device, model->main_device_id));
        model->weights.mlp_up_w[i] = wrap(create_tensor({interal, hidden}, meta->dtype, device, model->main_device_id));
        model->weights.mlp_down_w[i] = wrap(create_tensor({hidden, interal}, meta->dtype, device, model->main_device_id));

        // Cache: [maxseq, nkvh, head_dim]
        model->k_cache.push_back(create_tensor({meta->maxseq, meta->nkvh, head_dim}, meta->dtype, device, model->main_device_id));
        model->v_cache.push_back(create_tensor({meta->maxseq, meta->nkvh, head_dim}, meta->dtype, device, model->main_device_id));
    }

    return model;
}

void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    if (!model) {
        return;
    }
    auto w = model->weights;

    delete w.in_embed;
    delete w.out_embed;
    delete w.out_norm_w;

    for (size_t i = 0; i < model->meta.nlayer; ++i) {
        delete w.attn_norm_w[i];
        delete w.attn_q_w[i];
        delete w.attn_q_b[i];
        delete w.attn_k_w[i];
        delete w.attn_k_b[i];
        delete w.attn_v_w[i];
        delete w.attn_v_b[i];
        delete w.attn_o_w[i];
        delete w.mlp_norm_w[i];
        delete w.mlp_gate_w[i];
        delete w.mlp_up_w[i];
        delete w.mlp_down_w[i];
    }
    delete[] w.attn_norm_w;
    delete[] w.attn_q_w;
    delete[] w.attn_q_b;
    delete[] w.attn_k_w;
    delete[] w.attn_k_b;
    delete[] w.attn_v_w;
    delete[] w.attn_v_b;
    delete[] w.attn_o_w;
    delete[] w.mlp_norm_w;
    delete[] w.mlp_gate_w;
    delete[] w.mlp_up_w;
    delete[] w.mlp_down_w;

    delete model;
}

struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    return &model->weights;
}

int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    if (ntoken == 0) {
        return model->meta.end_token;
    }

    auto device = model->main_device;
    auto dev_id = model->main_device_id;
    auto dtype = model->meta.dtype;

    // 1. Input Embedding
    auto t_input_ids = create_tensor({ntoken}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
    std::memcpy(t_input_ids->data(), token_ids, ntoken * sizeof(int64_t));
    auto t_hidden_states = create_tensor({ntoken, model->meta.hs}, dtype, device, dev_id);
    ops::embedding(t_hidden_states, t_input_ids, unwrap(model->weights.in_embed));

    // 2. Pos IDs
    auto t_pos_ids = create_tensor({ntoken}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
    int64_t *pos_ptr = reinterpret_cast<int64_t *>(t_pos_ids->data());
    for (size_t i = 0; i < ntoken; ++i) {
        pos_ptr[i] = model->current_pos + i;
    }

    // 3. Layers
    size_t q_dim = model->meta.nh * model->meta.dh;
    size_t kv_dim = model->meta.nkvh * model->meta.dh;
    float scale = 1.0f / std::sqrt(static_cast<float>(model->meta.dh));

    for (size_t i = 0; i < model->meta.nlayer; ++i) {
        auto residual = t_hidden_states; // shared_ptr copy

        // RMS Norm
        auto t_normed = create_tensor({ntoken, model->meta.hs}, dtype, device, dev_id);
        ops::rms_norm(t_normed, t_hidden_states, unwrap(model->weights.attn_norm_w[i]), model->meta.epsilon);

        // QKV
        auto t_q = create_tensor({ntoken, q_dim}, dtype, device, dev_id);
        auto t_k = create_tensor({ntoken, kv_dim}, dtype, device, dev_id);
        auto t_v = create_tensor({ntoken, kv_dim}, dtype, device, dev_id);
        ops::linear(t_q, t_normed, unwrap(model->weights.attn_q_w[i]), unwrap(model->weights.attn_q_b[i]));
        ops::linear(t_k, t_normed, unwrap(model->weights.attn_k_w[i]), unwrap(model->weights.attn_k_b[i]));
        ops::linear(t_v, t_normed, unwrap(model->weights.attn_v_w[i]), unwrap(model->weights.attn_v_b[i]));

        auto t_q_3d = t_q->view({ntoken, model->meta.nh, model->meta.dh});
        auto t_k_3d = t_k->view({ntoken, model->meta.nkvh, model->meta.dh});
        auto t_v_3d = t_v->view({ntoken, model->meta.nkvh, model->meta.dh});

        // RoPE
        ops::rope(t_q_3d, t_q_3d, t_pos_ids, model->meta.theta);
        ops::rope(t_k_3d, t_k_3d, t_pos_ids, model->meta.theta);

        // Usage for KV Cache Update
        // Note: Assuming CPU and contiguous cache for simple memcpy update
        auto k_cache = model->k_cache[i];
        auto v_cache = model->v_cache[i];
        size_t offset = model->current_pos * kv_dim * k_cache->elementSize();
        size_t nbytes = ntoken * kv_dim * k_cache->elementSize();
        std::memcpy(k_cache->data() + offset, t_k_3d->data(), nbytes);
        std::memcpy(v_cache->data() + offset, t_v_3d->data(), nbytes);

        // Attention
        size_t total_len = model->current_pos + ntoken;
        // manually slice cache: view [total_len, nkvh, dh] - but Tensor::slice supports dim slicing
        // Our cache is [maxseq, nkvh, dh]. We want slice dim=0, 0 to total_len.
        auto k_ctx = k_cache->slice(0, 0, total_len);
        auto v_ctx = v_cache->slice(0, 0, total_len);

        auto t_attn_out = create_tensor({ntoken, model->meta.nh, model->meta.dh}, dtype, device, dev_id);
        ops::self_attention(t_attn_out, t_q_3d, k_ctx, v_ctx, scale);

        // O Proj
        auto t_attn_res = create_tensor({ntoken, model->meta.hs}, dtype, device, dev_id);
        auto t_attn_flat = t_attn_out->view({ntoken, q_dim});
        ops::linear(t_attn_res, t_attn_flat, unwrap(model->weights.attn_o_w[i]), nullptr);

        // Residual
        ops::add(t_hidden_states, residual, t_attn_res);
        residual = t_hidden_states;

        // FFN
        ops::rms_norm(t_normed, t_hidden_states, unwrap(model->weights.mlp_norm_w[i]), model->meta.epsilon);
        auto t_gate = create_tensor({ntoken, model->meta.di}, dtype, device, dev_id);
        auto t_up = create_tensor({ntoken, model->meta.di}, dtype, device, dev_id);
        ops::linear(t_gate, t_normed, unwrap(model->weights.mlp_gate_w[i]), nullptr);
        ops::linear(t_up, t_normed, unwrap(model->weights.mlp_up_w[i]), nullptr);

        auto t_swiglu = create_tensor({ntoken, model->meta.di}, dtype, device, dev_id);
        ops::swiglu(t_swiglu, t_gate, t_up);

        auto t_down = create_tensor({ntoken, model->meta.hs}, dtype, device, dev_id);
        ops::linear(t_down, t_swiglu, unwrap(model->weights.mlp_down_w[i]), nullptr);

        ops::add(t_hidden_states, residual, t_down);
    }

    // 4. Final
    auto t_final_norm = create_tensor({ntoken, model->meta.hs}, dtype, device, dev_id);
    ops::rms_norm(t_final_norm, t_hidden_states, unwrap(model->weights.out_norm_w), model->meta.epsilon);
    auto t_logits = create_tensor({ntoken, model->meta.voc}, dtype, device, dev_id);
    ops::linear(t_logits, t_final_norm, unwrap(model->weights.out_embed), nullptr);

    // 5. Argmax (last token)
    auto t_last_logit = t_logits->slice(0, ntoken - 1, ntoken); // [1, voc]
    auto t_last_logit_1d = t_last_logit->view({model->meta.voc});
    auto t_idx = create_tensor({1}, LLAISYS_DTYPE_I64, device, dev_id);
    auto t_val = create_tensor({1}, dtype, device, dev_id);
    ops::argmax(t_idx, t_val, t_last_logit_1d);

    int64_t next_token;
    std::memcpy(&next_token, t_idx->data(), sizeof(int64_t));
    model->current_pos += ntoken;

    return next_token;
}

} // extern "C"