#include "backend.h"
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace lora_omni {
namespace {
using Tensor = ggml_tensor;
Tensor * bias(DeviceGraph & g, Tensor * x, const std::string & name) {
    return ggml_add(g.ctx, x, g.weight(name));
}
Tensor * rms(DeviceGraph & g, Tensor * x, const std::string & name) {
    return ggml_mul(g.ctx, ggml_rms_norm(g.ctx, x, 1.e-5f), g.weight(name));
}
// Channels are contiguous in the public codec interface: [channels, time].
Tensor * conv(DeviceGraph & g, Tensor * x, const std::string & stem, int dilation = 1) {
    auto * c = g.ctx;
    auto * w = g.weight(stem + ".weight");
    const int64_t frames = x->ne[1], channels = x->ne[0], kernel = w->ne[0];
    auto * temporal = ggml_cont(c, ggml_transpose(c, x));
    temporal = ggml_pad_ext(c, temporal, int((kernel - 1) * dilation), 0, 0, 0, 0, 0, 0, 0);
    auto * columns = ggml_im2col(c, w, temporal, 1, 0, 0, 0, dilation, 0, false, GGML_TYPE_F32);
    columns = ggml_reshape_2d(c, columns, kernel * channels, frames);
    auto * result = ggml_mul_mat(c, ggml_reshape_2d(c, w, kernel * channels, w->ne[2]), columns);
    ggml_mul_mat_set_prec(result, GGML_PREC_F32);
    return bias(g, result, stem + ".bias");
}
Tensor * transpose_conv(DeviceGraph & g, Tensor * x, const std::string & stem, int stride, int crop) {
    auto * c = g.ctx;
    auto * w = g.weight(stem + ".weight");
    const int64_t kernel = w->ne[0], channels = w->ne[1], input_channels = w->ne[2];
    auto * matrix = ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, w, 1, 2, 0, 3)), input_channels, kernel * channels);
    auto * columns = ggml_mul_mat(c, matrix, x);
    ggml_mul_mat_set_prec(columns, GGML_PREC_F32);
    auto * result = ggml_col2im_1d(c, columns, stride, int(channels), crop);
    result = ggml_cont(c, ggml_transpose(c, result));
    return bias(g, result, stem + ".bias");
}
Tensor * snake(DeviceGraph & g, Tensor * x, const std::string & stem) {
    auto * c = g.ctx;
    auto * wave = ggml_sin(c, ggml_mul(c, x, ggml_exp(c, g.weight(stem + ".alpha"))));
    auto * denominator = ggml_add(c, ggml_exp(c, g.weight(stem + ".beta")), g.input({1.e-9f}, 1));
    return ggml_add(c, x, ggml_div(c, ggml_sqr(c, wave), denominator));
}
}
Vector device_codec_embedding(const std::string & path, const std::vector<Codes> & codes) {
    auto model = device_model(path, true);
    DeviceGraph g(*model);
    Tensor * sum = nullptr;
    for (int group = 0; group < 16; ++group) {
        std::vector<int32_t> ids;
        for (const auto & frame : codes) {
            if (frame[group] < 0 || frame[group] >= 2048) throw std::runtime_error("invalid codec ID");
            ids.push_back(group * 2048 + frame[group]);
        }
        auto * rows = ggml_get_rows(g.ctx, g.weight("code2wav.code_embedding.weight"), g.integers(ids));
        sum = sum ? ggml_add(g.ctx, sum, rows) : rows;
    }
    return g.run(ggml_scale(g.ctx, sum, 1.f / 16));
}
Vector device_codec_transformer(const std::string & path, const Vector & input) {
    auto model = device_model(path, true);
    Vector output = input;
    const int chunk = model->paged ? 1 : 8;
    for (int first = 0; first < 8; first += chunk) {
    DeviceGraph g(*model);
    auto * c = g.ctx;
    auto * x = g.input(output, 1024);
    const int frames = int(x->ne[1]);
    std::vector<int32_t> positions(frames);
    std::iota(positions.begin(), positions.end(), 0);
    auto * pos = g.integers(positions);
    Vector mask(size_t(frames) * frames);
    for (int t = 0; t < frames; ++t) for (int k = 0; k < frames; ++k)
        mask[size_t(t) * frames + k] = k <= t && k > t - 72 ? 0.f : -INFINITY;
    auto * attention_mask = g.input(mask, frames);
    for (int layer = first; layer < std::min(first + chunk, 8); ++layer) {
        const auto root = "code2wav.pre_transformer.layers." + std::to_string(layer) + ".";
        auto * n = rms(g, x, root + "input_layernorm.weight");
        auto * q = ggml_reshape_3d(c, g.linear(root + "self_attn.q_proj.weight", n), 64, 16, frames);
        auto * k = ggml_reshape_3d(c, g.linear(root + "self_attn.k_proj.weight", n), 64, 16, frames);
        auto * v = ggml_reshape_3d(c, g.linear(root + "self_attn.v_proj.weight", n), 64, 16, frames);
        q = ggml_rope_ext(c, q, pos, nullptr, 64, GGML_ROPE_TYPE_NEOX, 0, 10000.f, 1.f, 0.f, 1.f, 0.f, 0.f);
        k = ggml_rope_ext(c, k, pos, nullptr, 64, GGML_ROPE_TYPE_NEOX, 0, 10000.f, 1.f, 0.f, 1.f, 0.f, 0.f);
        auto * scores = ggml_mul_mat(c, ggml_permute(c, k, 0, 2, 1, 3), ggml_permute(c, q, 0, 2, 1, 3));
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        scores = ggml_soft_max_ext(c, scores, attention_mask, 1.f / 8, 0.f);
        auto * a = ggml_mul_mat(c, ggml_cont(c, ggml_permute(c, v, 1, 2, 0, 3)), scores);
        ggml_mul_mat_set_prec(a, GGML_PREC_F32);
        a = ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, a, 0, 2, 1, 3)), 1024, frames);
        a = ggml_mul(c, g.linear(root + "self_attn.o_proj.weight", a), g.weight(root + "self_attn_layer_scale.scale"));
        x = ggml_add(c, x, a);
        n = rms(g, x, root + "post_attention_layernorm.weight");
        auto * ffn = ggml_mul(c, ggml_silu(c, g.linear(root + "mlp.gate_proj.weight", n)), g.linear(root + "mlp.up_proj.weight", n));
        ffn = ggml_mul(c, g.linear(root + "mlp.down_proj.weight", ffn), g.weight(root + "mlp_layer_scale.scale"));
        x = ggml_add(c, x, ffn);
    }
    if (first + chunk >= 8) x = rms(g, x, "code2wav.pre_transformer.norm.weight");
    output = g.run(x);
    }
    return output;
}
Vector device_codec_upsample(const std::string & path, const Vector & input) {
    auto model = device_model(path, true);
    DeviceGraph g(*model);
    auto * c = g.ctx;
    auto * x = g.input(input, 1024);
    for (int stage = 0; stage < 2; ++stage) {
        const auto root = "code2wav.upsample." + std::to_string(stage) + ".";
        x = transpose_conv(g, x, root + "0.conv", 2, 0);
        const int frames = int(x->ne[1]);
        auto * temporal = ggml_cont(c, ggml_transpose(c, x));
        temporal = ggml_pad_ext(c, temporal, 6, 0, 0, 0, 0, 0, 0, 0);
        temporal = ggml_reshape_4d(c, temporal, frames + 6, 1, 1024, 1);
        auto * w = g.weight(root + "1.dwconv.conv.weight");
        auto * columns = ggml_im2col(c, w, temporal, 1, 0, 0, 0, 1, 0, false, GGML_TYPE_F32);
        auto * dw = ggml_mul_mat(c, w, columns);
        ggml_mul_mat_set_prec(dw, GGML_PREC_F32);
        dw = ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, dw, 2, 1, 0, 3)), 1024, frames);
        dw = bias(g, dw, root + "1.dwconv.conv.bias");
        auto * n = ggml_norm(c, dw, 1.e-6f);
        n = bias(g, ggml_mul(c, n, g.weight(root + "1.norm.weight")), root + "1.norm.bias");
        auto * pw = ggml_gelu_erf(c, bias(g, g.linear(root + "1.pwconv1.weight", n), root + "1.pwconv1.bias"));
        pw = bias(g, g.linear(root + "1.pwconv2.weight", pw), root + "1.pwconv2.bias");
        x = ggml_add(c, x, ggml_mul(c, pw, g.weight(root + "1.gamma")));
    }
    return g.run(x);
}
Vector device_codec_dac(const std::string & path, const Vector & input) {
    auto model = device_model(path, true);
    // Bound temporary memory by releasing the graph after each upsampling stage.
    Vector value;
    {
        DeviceGraph g(*model);
        value = g.run(conv(g, g.input(input, 1024), "code2wav.decoder.0.conv"));
    }
    const int strides[] = {8, 5, 4, 3};
    const int dilations[] = {1, 3, 9};
    int channels = 1536;
    for (int stage = 0; stage < 4; ++stage) {
        DeviceGraph g(*model);
        const auto root = "code2wav.decoder." + std::to_string(stage + 1) + ".block.";
        auto * x = snake(g, g.input(value, channels), root + "0");
        x = transpose_conv(g, x, root + "1.conv", strides[stage], strides[stage]);
        channels /= 2;
        for (int unit = 0; unit < 3; ++unit) {
            const auto stem = root + std::to_string(unit + 2) + ".";
            auto * y = conv(g, snake(g, x, stem + "act1"), stem + "conv1.conv", dilations[unit]);
            y = conv(g, snake(g, y, stem + "act2"), stem + "conv2.conv");
            x = ggml_add(g.ctx, x, y);
        }
        value = g.run(x);
    }
    DeviceGraph g(*model);
    auto * x = conv(g, snake(g, g.input(value, 96), "code2wav.decoder.5"), "code2wav.decoder.6.conv");
    return g.run(ggml_clamp(g.ctx, x, -1.f, 1.f));
}
}
