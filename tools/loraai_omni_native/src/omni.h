#pragma once
#include <array>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace lora_omni {
using Vector = std::vector<float>;
using Codes = std::array<int, 16>;
struct Prediction { Codes codes; Vector feedback; };
class Predictor {
public:
    Predictor(const std::string & model, int threads);
    ~Predictor();
    Prediction frame(const Vector & hidden, int code0, const Vector & embedding, std::mt19937 * rng = nullptr);
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
class Talker {
public:
    Talker(const std::string & model, int capacity, int threads);
    ~Talker();
    Vector step(const Vector & input);
    Vector logits(const Vector & hidden);
    Vector embedding(int id);
    void rewind(int prefix_length);
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
Vector project_text(const std::string & model, const Vector & input, int threads);
Vector project_hidden(const std::string & model, const Vector & input, int threads);
Vector codec_embedding(const std::string & model, const std::vector<Codes> & frames, int threads);
Vector codec_transformer(const std::string & model, const Vector & input, int threads);
Vector codec_upsample(const std::string & model, const Vector & input, int threads);
Vector codec_dac(const std::string & model, const Vector & input, int threads);
Vector decode_audio(const std::string & model, const std::vector<Codes> & frames, int threads);
int select_codec(const Vector & logits);
int sample_codec(const Vector & logits, std::mt19937 & rng, float temperature, float top_p, bool first_group);
}
