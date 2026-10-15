#include "omni.h"
#include "backend.h"
#include "codec_upsample.inc"
namespace lora_omni {
Vector codec_upsample(const std::string & model, const Vector & input, int threads) {
    if (input.empty() || input.size() % H) throw std::runtime_error("invalid upsample input");
    if (use_device_backend()) return device_codec_upsample(model, input);
    omni::Reader reader(model);
    Vector out = input;
    int frames = int(input.size() / H);
    for (int i = 0; i < 2; ++i) {
        Upsample block(reader, i);
        out = block.forward(out, frames, threads, nullptr);
        frames *= 2;
    }
    return out;
}
}
