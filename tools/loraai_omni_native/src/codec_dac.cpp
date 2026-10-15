#include "omni.h"
#include "backend.h"
#include "codec_dac.inc"
namespace lora_omni {
Vector codec_dac(const std::string & model, const Vector & input, int threads) {
    if (input.empty() || input.size() % 1024) throw std::runtime_error("invalid DAC input");
    if (use_device_backend()) return device_codec_dac(model, input);
    omni::Reader reader(model);
    int frames = int(input.size() / 1024);
    auto out = causal_conv(input, get(reader, "code2wav.decoder.0.conv.weight", {7, 1024, 1536}),
                           get(reader, "code2wav.decoder.0.conv.bias", {1536}), frames, 1024, 1536, 7, threads);
    const int strides[] = {8, 5, 4, 3};
    int channels = 1536;
    for (int i = 0; i < 4; ++i) {
        const std::string stem = "code2wav.decoder." + std::to_string(i + 1) + ".block.";
        out = snake(out, get(reader, stem + "0.alpha", {uint64_t(channels)}),
                    get(reader, stem + "0.beta", {uint64_t(channels)}), frames, channels);
        const int next = channels / 2, kernel = strides[i] * 2;
        auto trans = transpose(out, get(reader, stem + "1.conv.weight", {uint64_t(kernel), uint64_t(next), uint64_t(channels)}),
                              get(reader, stem + "1.conv.bias", {uint64_t(next)}), frames, channels, next, kernel, strides[i], threads);
        out = std::move(trans.cropped);
        frames = output_frames(frames, strides[i], kernel);
        channels = next;
        for (int u = 0; u < 3; ++u) out = unit(out, read_unit(reader, u, channels, i + 1), frames, channels, DILATIONS[u], threads).residual;
    }
    out = snake(out, get(reader, "code2wav.decoder.5.alpha", {96}), get(reader, "code2wav.decoder.5.beta", {96}), frames, 96);
    out = causal_conv(out, get(reader, "code2wav.decoder.6.conv.weight", {7, 96, 1}), get(reader, "code2wav.decoder.6.conv.bias", {1}), frames, 96, 1, 7, threads);
    for (float & value : out) value = std::max(-1.f, std::min(1.f, value));
    return out;
}
}
