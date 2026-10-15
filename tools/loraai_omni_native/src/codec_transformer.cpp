#include "omni.h"
#include "backend.h"
#include "codec_transformer.inc"
namespace lora_omni {
Vector codec_transformer(const std::string & model, const Vector & input, int threads) {
    if (input.empty() || input.size() % H || input.size() > H * 512) throw std::runtime_error("invalid codec transformer input");
    if (use_device_backend()) return device_codec_transformer(model, input);
    omni::Reader reader(model);
    std::vector<Block> layers;
    for (int i = 0; i < 8; ++i) layers.emplace_back(reader, i);
    const auto gain = get(reader, std::string(PREFIX) + "norm.weight", {H});
    Vector out;
    out.reserve(input.size());
    for (size_t i = 0; i < input.size() / H; ++i) {
        Vector value(input.begin() + i * H, input.begin() + (i + 1) * H);
        for (auto & layer : layers) value = layer.forward(value, int(i), 10000, threads);
        value = norm(value, gain);
        out.insert(out.end(), value.begin(), value.end());
    }
    return out;
}
}
