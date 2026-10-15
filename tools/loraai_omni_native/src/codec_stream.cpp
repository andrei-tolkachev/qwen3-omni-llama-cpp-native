#include "codec_stream.h"
#include "backend.h"
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace lora_omni {
Vector CodecStream::transform(const Vector & input) {
    auto weights = device_model(model, true);
    Vector output = input;
    const int count = int(input.size() / 1024);
    for (int layer = 0; layer < 8; ++layer) {
        DeviceGraph g(*weights);
        auto * c = g.ctx;
        auto * x = g.input(output, 1024);
        const auto root = "code2wav.pre_transformer.layers." + std::to_string(layer) + ".";
        auto norm = [&](ggml_tensor * t, const std::string & name) {
            return ggml_mul(c, ggml_rms_norm(c, t, 1.e-5f), g.weight(name));
        };
        auto * n = norm(x, root + "input_layernorm.weight");
        auto * q = ggml_reshape_3d(c, g.linear(root + "self_attn.q_proj.weight", n), 64, 16, count);
        auto * k = ggml_reshape_3d(c, g.linear(root + "self_attn.k_proj.weight", n), 64, 16, count);
        auto * v = ggml_reshape_3d(c, g.linear(root + "self_attn.v_proj.weight", n), 64, 16, count);
        std::vector<int32_t> positions(count);
        std::iota(positions.begin(), positions.end(), position);
        auto * pos = g.integers(positions);
        q = ggml_rope_ext(c, q, pos, nullptr, 64, GGML_ROPE_TYPE_NEOX, 0, 10000.f, 1.f, 0.f, 1.f, 0.f, 0.f);
        k = ggml_rope_ext(c, k, pos, nullptr, 64, GGML_ROPE_TYPE_NEOX, 0, 10000.f, 1.f, 0.f, 1.f, 0.f, 0.f);
        const int past = int(keys[layer].size() / 1024), total = past + count;
        if (past) {
            k = ggml_concat(c, ggml_reshape_3d(c, g.input(keys[layer], 1024), 64, 16, past), k, 2);
            v = ggml_concat(c, ggml_reshape_3d(c, g.input(values[layer], 1024), 64, 16, past), v, 2);
        }
        Vector mask(size_t(total) * count);
        for (int t = 0; t < count; ++t) for (int j = 0; j < total; ++j)
            mask[size_t(t) * total + j] = j <= past + t && j > past + t - 72 ? 0.f : -INFINITY;
        auto * scores = ggml_mul_mat(c, ggml_permute(c, k, 0, 2, 1, 3), ggml_permute(c, q, 0, 2, 1, 3));
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        scores = ggml_soft_max_ext(c, scores, g.input(mask, total), 1.f / 8, 0.f);
        auto * a = ggml_mul_mat(c, ggml_cont(c, ggml_permute(c, v, 1, 2, 0, 3)), scores);
        ggml_mul_mat_set_prec(a, GGML_PREC_F32);
        a = ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, a, 0, 2, 1, 3)), 1024, count);
        x = ggml_add(c, x, ggml_mul(c, g.linear(root + "self_attn.o_proj.weight", a), g.weight(root + "self_attn_layer_scale.scale")));
        n = norm(x, root + "post_attention_layernorm.weight");
        auto * ffn = ggml_mul(c, ggml_silu(c, g.linear(root + "mlp.gate_proj.weight", n)), g.linear(root + "mlp.up_proj.weight", n));
        x = ggml_add(c, x, ggml_mul(c, g.linear(root + "mlp.down_proj.weight", ffn), g.weight(root + "mlp_layer_scale.scale")));
        if (layer == 7) x = norm(x, "code2wav.pre_transformer.norm.weight");
        auto * packed = ggml_concat(c, ggml_reshape_1d(c, x, count * 1024), ggml_reshape_1d(c, ggml_cont(c, k), total * 1024), 0);
        packed = ggml_concat(c, packed, ggml_reshape_1d(c, ggml_cont(c, v), total * 1024), 0);
        auto result = g.run(packed);
        output.assign(result.begin(), result.begin() + count * 1024);
        const int keep = std::min(71, total);
        keys[layer].assign(result.begin() + (count + total - keep) * 1024, result.begin() + (count + total) * 1024);
        values[layer].assign(result.end() - keep * 1024, result.end());
    }
    position += count;
    return output;
}
Vector CodecStream::push(const std::vector<Codes> & codes) {
    if (codes.empty() || codes.size() > 16 || !use_device_backend()) throw std::runtime_error("invalid GPU codec stream input");
    auto current = transform(device_codec_embedding(model, codes));
    const size_t old_up = up_history.size();
    up_history.insert(up_history.end(), current.begin(), current.end());
    auto up = device_codec_upsample(model, up_history);
    up.erase(up.begin(), up.begin() + old_up * 4);
    if (up_history.size() > 16 * 1024) up_history.erase(up_history.begin(), up_history.end() - 16 * 1024);
    const size_t past_latents = dac_history.size() / 1024;
    const size_t start_latent = size_t(position - codes.size()) * 4 - past_latents;
    dac_history.insert(dac_history.end(), up.begin(), up.end());
    auto wave = device_codec_dac(model, dac_history);
    // DAC stride is 8*5*4*3=480. The fixed end crop is 555 samples.
    // 32 latent rows cover its full left receptive field; retained attention
    // and 16 upstream rows preserve state without crossfading or inserted gaps.
    const size_t offset = emitted - start_latent * 480;
    if (offset > wave.size()) throw std::runtime_error("invalid codec stream offset");
    Vector pcm(wave.begin() + offset, wave.end());
    emitted += pcm.size();
    if (dac_history.size() > 32 * 1024) dac_history.erase(dac_history.begin(), dac_history.end() - 32 * 1024);
    release_unused_device_workspaces();
    return pcm;
}
}
