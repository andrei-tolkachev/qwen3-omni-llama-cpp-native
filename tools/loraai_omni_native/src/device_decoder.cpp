#include "backend.h"
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace lora_omni {
DeviceDecoder::DeviceDecoder(const std::string & path, int capacity, bool predictor):
    model(device_model(path)), max_capacity(capacity), layers(predictor ? 5 : 20),
    kv_heads(predictor ? 8 : 2), predictor(predictor),
    prefix(predictor ? "talker.code_predictor.model." : "talker.model.") {
    if (capacity < 1 || capacity > 65536) throw std::runtime_error("invalid Omni model context capacity");
    grow(std::min(capacity, 512));
}
void DeviceDecoder::grow(int required) {
    if (required <= capacity) return;
    if (required > max_capacity) throw std::runtime_error("voice_talker_context_exhausted");
    const int next = std::min(max_capacity, std::max(required, std::max(512, capacity * 2)));
    // The old KV stays valid until the complete replacement has been copied.
    ensure_device_memory(size_t(layers) * 2 * 128 * kv_heads * next * sizeof(float));
    auto * context = ggml_init({size_t(layers * 4 + 1) * ggml_tensor_overhead(), nullptr, true});
    if (!context) throw std::runtime_error("cannot create Omni KV context");
    std::vector<ggml_tensor *> nk, nv;
    for (int i = 0; i < layers; ++i) {
        nk.push_back(ggml_new_tensor_3d(context, GGML_TYPE_F32, 128, kv_heads, next));
        nv.push_back(ggml_new_tensor_3d(context, GGML_TYPE_F32, 128, kv_heads, next));
    }
    auto buffer = ggml_backend_alloc_ctx_tensors(context, model->backend);
    if (!buffer) { ggml_free(context); throw std::runtime_error("cannot allocate Omni GPU KV memory"); }
    ggml_backend_synchronize(model->backend);
    for (int i = 0; i < layers && capacity; ++i) {
        auto * key_view = ggml_view_3d(context, nk[i], 128, kv_heads, capacity, nk[i]->nb[1], nk[i]->nb[2], 0);
        auto * value_view = ggml_view_3d(context, nv[i], 128, kv_heads, capacity, nv[i]->nb[1], nv[i]->nb[2], 0);
        if (ggml_backend_view_init(key_view) != GGML_STATUS_SUCCESS || ggml_backend_view_init(value_view) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buffer); ggml_free(context);
            throw std::runtime_error("Omni GPU KV view initialization failed");
        }
        ggml_backend_tensor_copy(keys[i], key_view);
        ggml_backend_tensor_copy(values[i], value_view);
    }
    ggml_backend_synchronize(model->backend);
    ggml_backend_buffer_free(cache_buffer); ggml_free(cache_context);
    cache_buffer = buffer; cache_context = context; keys = std::move(nk); values = std::move(nv); capacity = next;
    std::fprintf(stderr, "Omni KV: %d positions, %zu bytes\n", capacity, ggml_backend_buffer_get_size(buffer));
}
DeviceDecoder::~DeviceDecoder() {
    ggml_backend_buffer_free(cache_buffer);
    ggml_free(cache_context);
}
void DeviceDecoder::reset() { position = 0; }
Vector DeviceDecoder::step(const Vector & input) {
    grow(position + 1);
    if (input.size() != 1024) throw std::runtime_error("Omni decoder input or cache capacity invalid");
    Vector result = input;
    const int chunk = model->paged ? 1 : layers;
    for (int first = 0; first < layers; first += chunk) {
    DeviceGraph g(*model);
    auto * c = g.ctx;
    auto * x = g.input(result, 1024);
    auto * pos = g.integers({position});
    for (int i = first; i < std::min(layers, first + chunk); ++i) {
        const std::string root = prefix + "layers." + std::to_string(i) + ".";
        auto * n = g.norm(root + "input_layernorm.weight", x);
        auto * q = ggml_reshape_3d(c, g.linear(root + "self_attn.q_proj.weight", n), 128, 16, 1);
        auto * k = ggml_reshape_3d(c, g.linear(root + "self_attn.k_proj.weight", n), 128, kv_heads, 1);
        auto * v = ggml_reshape_3d(c, g.linear(root + "self_attn.v_proj.weight", n), 128, kv_heads, 1);
        q = g.norm(root + "self_attn.q_norm.weight", q);
        k = g.norm(root + "self_attn.k_norm.weight", k);
        // Talker text/audio positions use equal M-RoPE axes, equivalent to NEOX here.
        q = ggml_rope_ext(c, q, pos, nullptr, 128, GGML_ROPE_TYPE_NEOX, 0, 1000000.f, 1.f, 0.f, 1.f, 0.f, 0.f);
        k = ggml_rope_ext(c, k, pos, nullptr, 128, GGML_ROPE_TYPE_NEOX, 0, 1000000.f, 1.f, 0.f, 1.f, 0.f, 0.f);
        const size_t offset = size_t(position) * 128 * kv_heads * sizeof(float);
        ggml_build_forward_expand(g.graph, ggml_cpy(c, k, ggml_view_3d(c, keys[i], 128, kv_heads, 1, keys[i]->nb[1], keys[i]->nb[2], offset)));
        ggml_build_forward_expand(g.graph, ggml_cpy(c, v, ggml_view_3d(c, values[i], 128, kv_heads, 1, values[i]->nb[1], values[i]->nb[2], offset)));
        auto * kh = ggml_view_3d(c, keys[i], 128, kv_heads, position + 1, keys[i]->nb[1], keys[i]->nb[2], 0);
        auto * vh = ggml_view_3d(c, values[i], 128, kv_heads, position + 1, values[i]->nb[1], values[i]->nb[2], 0);
        q = ggml_permute(c, q, 0, 2, 1, 3);
        kh = ggml_permute(c, kh, 0, 2, 1, 3);
        vh = ggml_cont(c, ggml_permute(c, vh, 1, 2, 0, 3));
        auto * scores = ggml_mul_mat(c, kh, q);
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        scores = ggml_soft_max(c, ggml_scale(c, scores, 1.f / std::sqrt(128.f)));
        auto * attention = ggml_mul_mat(c, vh, scores);
        ggml_mul_mat_set_prec(attention, GGML_PREC_F32);
        attention = ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, attention, 0, 2, 1, 3)), 2048, 1);
        x = ggml_add(c, x, g.linear(root + "self_attn.o_proj.weight", attention));
        n = g.norm(root + "post_attention_layernorm.weight", x);
        auto dense = [&](const std::string & stem) {
            return g.linear(stem + "down_proj.weight", ggml_mul(c,
                ggml_silu(c, g.linear(stem + "gate_proj.weight", n)), g.linear(stem + "up_proj.weight", n)));
        };
        ggml_tensor * ffn;
        if (predictor) ffn = dense(root + "mlp.");
        else {
            auto * probs = ggml_soft_max(c, g.linear(root + "mlp.gate.weight", n));
            auto * ids = ggml_argsort_top_k(c, probs, 6);
            auto * weights = ggml_get_rows(c, ggml_reshape_3d(c, probs, 1, 128, 1), ids);
            weights = ggml_reshape_2d(c, weights, 6, 1);
            weights = ggml_div(c, weights, ggml_sum_rows(c, weights));
            if (model->paged) {
                // Routing stays on the GPU; transfer only the six selected expert slices.
                auto * state = ggml_concat(c, x, n, 0);
                state = ggml_concat(c, state, weights, 0);
                state = ggml_concat(c, state, ggml_cast(c, ids, GGML_TYPE_F32), 0);
                const auto routed_state = g.run(state);
                std::vector<int32_t> selected(6);
                for (int j = 0; j < 6; ++j) selected[j] = int32_t(routed_state[2054 + j]);
                model->release_pages();
                DeviceGraph h(*model);
                auto * hc = h.ctx;
                auto * residual = h.input(Vector(routed_state.begin(), routed_state.begin() + 1024), 1024);
                auto * normalized = h.input(Vector(routed_state.begin() + 1024, routed_state.begin() + 2048), 1024);
                auto * probabilities = h.input(Vector(routed_state.begin() + 2048, routed_state.begin() + 2054), 6);
                probabilities = ggml_reshape_3d(hc, probabilities, 1, 6, 1);
                auto * local_ids = h.integers({0, 1, 2, 3, 4, 5});
                auto * seed = ggml_reshape_3d(hc, normalized, 1024, 1, 1);
                const auto stem = root + "mlp.experts_packed.";
                auto * gate = ggml_mul_mat_id(hc, model->experts(hc, stem + "gate_proj.weight", selected), seed, local_ids);
                auto * up = ggml_mul_mat_id(hc, model->experts(hc, stem + "up_proj.weight", selected), seed, local_ids);
                auto * routed = ggml_mul_mat_id(hc, model->experts(hc, stem + "down_proj.weight", selected), ggml_mul(hc, ggml_silu(hc, gate), up), local_ids);
                routed = ggml_mul(hc, routed, probabilities);
                auto * sum = ggml_view_2d(hc, routed, 1024, 1, routed->nb[2], 0);
                for (int j = 1; j < 6; ++j) sum = ggml_add(hc, sum, ggml_view_2d(hc, routed, 1024, 1, routed->nb[2], j * routed->nb[1]));
                const auto shared_root = root + "mlp.shared_expert.";
                auto * shared = h.linear(shared_root + "down_proj.weight", ggml_mul(hc,
                    ggml_silu(hc, h.linear(shared_root + "gate_proj.weight", normalized)), h.linear(shared_root + "up_proj.weight", normalized)));
                shared = ggml_mul(hc, shared, ggml_sigmoid(hc, h.linear(root + "mlp.shared_expert_gate.weight", normalized)));
                auto * output = ggml_add(hc, residual, ggml_add(hc, sum, shared));
                if (i == layers - 1) output = h.norm(prefix + "norm.weight", output);
                result = h.run(output);
                x = nullptr;
                continue;
            }
            weights = ggml_reshape_3d(c, weights, 1, 6, 1);
            auto * seed = ggml_reshape_3d(c, n, 1024, 1, 1);
            const std::string stem = root + "mlp.experts_packed.";
            auto * gate = ggml_mul_mat_id(c, g.weight(stem + "gate_proj.weight"), seed, ids);
            auto * up = ggml_mul_mat_id(c, g.weight(stem + "up_proj.weight"), seed, ids);
            auto * routed = ggml_mul_mat_id(c, g.weight(stem + "down_proj.weight"), ggml_mul(c, ggml_silu(c, gate), up), ids);
            routed = ggml_mul(c, routed, weights);
            ffn = ggml_view_2d(c, routed, 1024, 1, routed->nb[2], 0);
            for (int j = 1; j < 6; ++j)
                ffn = ggml_add(c, ffn, ggml_view_2d(c, routed, 1024, 1, routed->nb[2], j * routed->nb[1]));
            auto * shared = dense(root + "mlp.shared_expert.");
            shared = ggml_mul(c, shared, ggml_sigmoid(c, g.linear(root + "mlp.shared_expert_gate.weight", n)));
            ffn = ggml_add(c, ffn, shared);
        }
        x = ggml_add(c, x, ffn);
    }
    if (x) {
        if (!predictor && first + chunk >= layers) x = g.norm(prefix + "norm.weight", x);
        result = g.run(x);
    }
    }
    ++position;
    return result;
}
Vector DeviceDecoder::logits(const Vector & input, int group) {
    DeviceGraph g(*model);
    auto * x = g.input(input, 1024);
    if (predictor) {
        if (group < 0 || group >= 15) throw std::runtime_error("invalid predictor head");
        return g.run(g.linear("talker.code_predictor.lm_head." + std::to_string(group) + ".weight", g.norm(prefix + "norm.weight", x)));
    }
    return g.run(g.linear("talker.codec_head.weight", x));
}
Vector DeviceDecoder::embedding(int id, int group) {
    if (id < 0 || id >= (predictor ? 2048 : 3072) || (predictor && (group < 0 || group >= 15)))
        throw std::runtime_error("invalid device codec embedding index");
    DeviceGraph g(*model);
    const auto name = prefix + "codec_embedding." + (predictor ? std::to_string(group) + "." : "") + "weight";
    return g.run(ggml_get_rows(g.ctx, g.weight(name), g.integers({id})));
}
Vector device_project(const std::string & path, const Vector & input, const std::string & mode) {
    auto model = device_model(path);
    DeviceGraph g(*model);
    const std::string stem = "talker." + mode + "_projection.";
    auto * x = g.input(input, 2048);
    x = ggml_silu(g.ctx, ggml_add(g.ctx, g.linear(stem + "linear_fc1.weight", x), g.weight(stem + "linear_fc1.bias")));
    return g.run(ggml_add(g.ctx, g.linear(stem + "linear_fc2.weight", x), g.weight(stem + "linear_fc2.bias")));
}
}
