#include "omni.h"
#include "backend.h"
#include "codec_embedding.inc"
namespace lora_omni {
Vector codec_embedding(const std::string & model, const std::vector<Codes> & codes, int threads) {
    if (codes.empty() || codes.size() > 512) throw std::runtime_error("invalid codec frame count");
    if (use_device_backend()) return device_codec_embedding(model, codes);
    Frames frames(codes.size());
    for (size_t i = 0; i < codes.size(); ++i) for (int j = 0; j < 16; ++j) {
        if (codes[i][j] < 0 || codes[i][j] >= 2048) throw std::runtime_error("invalid codec ID");
        frames[i][j] = uint16_t(codes[i][j]);
    }
    omni::Reader reader(model);
    model_preflight(reader);
    return ggml_mean_all(bf16_embedding_rows(reader, frames), int(frames.size()), threads);
}
Vector decode_audio(const std::string & model, const std::vector<Codes> & frames, int threads) {
    return codec_dac(model, codec_upsample(model, codec_transformer(model, codec_embedding(model, frames, threads), threads), threads), threads);
}
}
