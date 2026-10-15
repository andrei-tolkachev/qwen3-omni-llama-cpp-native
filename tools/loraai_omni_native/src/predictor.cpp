#include "omni.h"
#include "backend.h"
#include "predictor.inc"

namespace lora_omni {
struct Predictor::Impl {
    std::unique_ptr<DeviceDecoder> device;
    omni::Reader reader;
    std::vector<std::unique_ptr<ARBlock>> blocks;
    Vector norm;
    std::vector<Vector> heads;
    int threads;
    Impl(const std::string & model, int nthreads): reader(model), threads(nthreads) {
        if (use_device_backend()) { device = std::make_unique<DeviceDecoder>(model, 16, true); return; }
        for (int i = 0; i < LAYERS; ++i) blocks.push_back(std::make_unique<ARBlock>(reader, i));
        norm = cp_weight(reader, "talker.code_predictor.model.norm.weight", {H});
        for (int i = 0; i < 15; ++i)
            heads.push_back(cp_weight(reader, "talker.code_predictor.lm_head." + std::to_string(i) + ".weight", {H, CODEVOCAB}));
    }
    Vector forward(Vector value, int position) {
        for (auto & block : blocks) value = block->forward(value, position, threads);
        return value;
    }
};
Predictor::Predictor(const std::string & model, int threads): impl(new Impl(model, threads)) {}
Predictor::~Predictor() = default;
Prediction Predictor::frame(const Vector & hidden, int code0, const Vector & embedding, std::mt19937 * rng) {
    if (code0 < 0 || code0 >= 2048 || embedding.size() != H || hidden.size() != H)
        throw std::runtime_error("invalid predictor seed");
    if (impl->device) {
        auto & device = *impl->device;
        device.reset();
        device.step(hidden);
        Vector last = device.step(embedding);
        Prediction out{};
        out.codes[0] = code0;
        std::vector<double> sum(embedding.begin(), embedding.end());
        for (int group = 1; group < 16; ++group) {
            if (group > 1) last = device.step(device.embedding(out.codes[group - 1], group - 2));
            const auto logits = device.logits(last, group - 1);
            const int id = rng ? sample_codec(logits, *rng, 1.f, .8f, false) : int(std::max_element(logits.begin(), logits.end()) - logits.begin());
            out.codes[group] = id;
            const auto row = device.embedding(id, group - 1);
            for (int j = 0; j < H; ++j) sum[j] += row[j];
        }
        out.feedback.assign(sum.begin(), sum.end());
        return out;
    }
    for (auto & block : impl->blocks) { block->key_cache.clear(); block->value_cache.clear(); }
    impl->forward(hidden, 0);
    Vector last = impl->forward(embedding, 1);
    Prediction out{};
    out.codes[0] = code0;
    std::vector<double> sum(embedding.begin(), embedding.end());
    for (int group = 1; group < 16; ++group) {
        if (group > 1) last = impl->forward(ar_lookup(impl->reader, group - 2, out.codes[group - 1]), group);
        Vector normalized = last;
        cp_rms(normalized, impl->norm, H);
        const auto logits = cp_mul(impl->heads[group - 1], normalized, H, CODEVOCAB, impl->threads);
        const int id = rng ? sample_codec(logits, *rng, 1.f, .8f, false) : int(std::max_element(logits.begin(), logits.end()) - logits.begin());
        out.codes[group] = id;
        const auto row = ar_lookup(impl->reader, group - 1, id);
        for (int j = 0; j < H; ++j) sum[j] += row[j];
    }
    out.feedback.assign(sum.begin(), sum.end());
    return out;
}
}
