#include "omni.h"
#include "backend.h"
#include "talker.inc"

namespace lora_omni {
struct Talker::Impl {
    std::unique_ptr<DeviceDecoder> device;
    omni::Reader reader;
    std::vector<std::unique_ptr<LayerB6>> layers;
    Vector norm, head;
    int position = 0, threads;
    Impl(const std::string & model, int capacity, int nthreads): reader(model), threads(nthreads) {
        if (use_device_backend()) { device = std::make_unique<DeviceDecoder>(model, capacity, false); return; }
        for (int i = 0; i < 20; ++i) layers.push_back(std::make_unique<LayerB6>(reader, i, capacity, threads));
        norm = weight_b6(reader, "talker.model.norm.weight", 1024);
        head = weight_b6(reader, "talker.codec_head.weight", 1024, 3072);
    }
};
Talker::Talker(const std::string & model, int capacity, int threads): impl(new Impl(model, capacity, threads)) {}
Talker::~Talker() = default;
void Talker::rewind(int prefix_length) {
    if (!impl->device) throw std::runtime_error("Talker prefix reuse requires device backend");
    impl->device->rewind(prefix_length);
}
Vector Talker::step(const Vector & input) {
    if (impl->device) return impl->device->step(input);
    Vector value = input;
    for (auto & layer : impl->layers) value = layer->eval(impl->position, value).output;
    ++impl->position;
    omni_attn::rms_norm_inplace(value.data(), impl->norm.data(), 1024);
    return value;
}
Vector Talker::logits(const Vector & hidden) { if (impl->device) return impl->device->logits(hidden); return mul_b6(impl->head, hidden, 1024, 3072, impl->threads); }
Vector Talker::embedding(int id) {
    if (impl->device) return impl->device->embedding(id);
    if (id < 0 || id >= 3072) throw std::runtime_error("invalid codec embedding ID");
    const auto & tensor = impl->reader.get_tensor("talker.model.codec_embedding.weight");
    if (tensor.type != omni::GGUF_BF16 || tensor.dims != std::vector<std::uint64_t>{1024, 3072})
        throw std::runtime_error("invalid Talker codec embedding shape");
    return to_f32(impl->reader.read_region(tensor.offset + std::uint64_t(id) * 2048, 2048));
}
int select_codec(const Vector & logits) {
    if (logits.size() != 3072) throw std::runtime_error("invalid Talker logits");
    int best = -1;
    for (int i = 0; i < 3072; ++i) {
        if (!std::isfinite(logits[i])) throw std::runtime_error("nonfinite Talker logits");
        if (i >= 2048 && i != 2150) continue;
        if (best < 0 || logits[i] > logits[best]) best = i;
    }
    return best;
}
int sample_codec(const Vector & logits, std::mt19937 & rng, float temperature, float top_p, bool first_group) {
    std::vector<int> ids;
    for (size_t i = 0; i < logits.size(); ++i) {
        if (!std::isfinite(logits[i])) throw std::runtime_error("nonfinite codec logits");
        if (first_group && i >= 2048 && i != 2150) continue;
        ids.push_back(int(i));
    }
    if (ids.empty()) throw std::runtime_error("empty codec vocabulary");
    const size_t count = std::min(size_t(50), ids.size());
    std::partial_sort(ids.begin(), ids.begin() + count, ids.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    ids.resize(count);
    std::vector<double> weights;
    double total = 0;
    for (int id : ids) { weights.push_back(std::exp(double(logits[id] - logits[ids[0]]) / temperature)); total += weights.back(); }
    double cumulative = 0;
    for (size_t i = 0; i < weights.size(); ++i) {
        cumulative += weights[i];
        if (cumulative >= top_p * total) { weights.resize(i + 1); break; }
    }
    return ids[std::discrete_distribution<size_t>(weights.begin(), weights.end())(rng)];
}
}
