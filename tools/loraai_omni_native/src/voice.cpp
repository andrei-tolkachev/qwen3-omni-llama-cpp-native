#include "voice.h"
#include "audio_join.h"
#include "codec_stream.h"
#include "backend.h"
#include "gguf.h"
#include "ggml.h"
#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <stdexcept>

namespace lora_omni {
namespace {
class Embeddings {
    gguf_context * metadata;
    std::ifstream file;
    size_t offset, bytes;
    ggml_type type;
    int64_t count;
public:
    explicit Embeddings(const std::string & model): file(model, std::ios::binary) {
        metadata = gguf_init_from_file(model.c_str(), {true, nullptr});
        if (!metadata) throw std::runtime_error("cannot read Thinker GGUF metadata");
        const auto tensor = gguf_find_tensor(metadata, "token_embd.weight");
        if (tensor < 0 || gguf_get_tensor_ne(metadata, tensor)[0] != 2048) {
            gguf_free(metadata); throw std::runtime_error("unsupported Thinker embedding geometry");
        }
        count = gguf_get_tensor_ne(metadata, tensor)[1];
        offset = gguf_get_data_offset(metadata) + gguf_get_tensor_offset(metadata, tensor);
        type = gguf_get_tensor_type(metadata, tensor);
        bytes = ggml_row_size(type, 2048);
    }
    ~Embeddings() { gguf_free(metadata); }
    Vector row(int id) {
        if (id < 0 || id >= count) throw std::runtime_error("invalid text token");
        std::vector<char> data(bytes);
        file.seekg(offset + size_t(id) * bytes);
        file.read(data.data(), bytes);
        if (!file) throw std::runtime_error("cannot read token embedding");
        Vector value(2048);
        if (type == GGML_TYPE_F32) std::memcpy(value.data(), data.data(), 8192);
        else {
            const auto traits = ggml_get_type_traits(type);
            if (!traits->to_float) throw std::runtime_error("unsupported embedding quantization");
            traits->to_float(data.data(), value.data(), 2048);
        }
        return value;
    }
};
void append(Vector & target, const Vector & value) { target.insert(target.end(), value.begin(), value.end()); }
Vector row(const Vector & value, size_t i) { return Vector(value.begin() + i * 1024, value.begin() + (i + 1) * 1024); }
std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    int n = -llama_tokenize(vocab, text.data(), int(text.size()), nullptr, 0, false, false);
    if (n < 1) throw std::runtime_error("voice_text_empty");
    if (n > 65536) throw std::runtime_error("speech text token limit exceeded");
    std::vector<llama_token> tokens(n);
    if (llama_tokenize(vocab, text.data(), int(text.size()), tokens.data(), n, false, false) != n)
        throw std::runtime_error("speech tokenization failed");
    return tokens;
}
}
std::string talker_model_path(const std::string & model) {
    if (const auto * path = std::getenv("OMNI_TALKER_MODEL")) {
        if (!*path) throw std::runtime_error("OMNI_TALKER_MODEL must not be empty");
        return path;
    }
    const auto models = std::getenv("OMNI_SPEECH_MODELS") ? std::filesystem::path(std::getenv("OMNI_SPEECH_MODELS")) : std::filesystem::path(model).parent_path() / "omni-experimental";
    const auto q4 = models / "talker-experts-packed-Q4.gguf";
    if (std::filesystem::is_regular_file(q4)) return q4.string();
    std::fprintf(stderr, "Omni: Q4 Talker not found; using BF16. A CUDA expert-matmul stall is known on one BF16 input; see README.\n");
    return (models / "talker-experts-packed-bf16-staging.gguf").string();
}
static VoiceResult speak_part(const std::string & model, const std::vector<CapturedRow> & captured,
                  const std::vector<llama_token> & tokens, int speaker, int max_frames, int threads,
                  const std::function<bool()> & stop, std::optional<uint32_t> seed,
                  const std::function<void(const Vector &)> & emit_audio) {
    if (speaker < 2301 || speaker > 2303 || max_frames < 1 || max_frames > 65536 || captured.size() > 4096)
        throw std::runtime_error("invalid speech limits");
    DeviceCancellationScope cancellation(stop);
    if (stop()) throw std::runtime_error("voice_cancelled");
    const auto models = std::getenv("OMNI_SPEECH_MODELS") ? std::filesystem::path(std::getenv("OMNI_SPEECH_MODELS")) : std::filesystem::path(model).parent_path() / "omni-experimental";
    const auto talker_model = talker_model_path(model);
    Embeddings embedding(model);
    Vector text_inputs, hidden_inputs;
    std::vector<bool> multimodal;
    size_t conditioning_start = 0, last_user = 0, conditioning_rows = 0;
    bool counting_user = false;
    for (size_t i = 0; i < captured.size(); ++i) {
        if (captured[i].token == 151644) {
            counting_user = i + 1 < captured.size() && captured[i + 1].token == 872;
            if (counting_user) last_user = i;
        }
        if (counting_user) ++conditioning_rows;
    }
    // Keep Thinker's full dialogue context. If speech conditioning would exceed
    // Talker's limit, use the latest user turn (whose hidden states already
    // attend to that dialogue) instead of failing or moving speech to CPU.
    if (conditioning_rows + 9 + tokens.size() > 65536) conditioning_start = last_user;
    bool in_user = false;
    for (size_t i = conditioning_start; i < captured.size(); ++i) {
        const auto & entry = captured[i];
        if (entry.token == 151644) in_user = i + 1 < captured.size() && captured[i + 1].token == 872;
        if (!in_user) continue;
        multimodal.push_back(entry.token < 0);
        if (entry.token < 0) {
            if (entry.hidden.size() != 2048) throw std::runtime_error("missing multimodal Thinker state");
            append(hidden_inputs, entry.hidden);
        } else append(text_inputs, embedding.row(entry.token));
    }
    if (multimodal.empty()) throw std::runtime_error("no uncached user conditioning captured");
    auto text_projection = project_text(talker_model, text_inputs, threads);
    auto hidden_projection = hidden_inputs.empty() ? Vector{} : project_hidden(talker_model, hidden_inputs, threads);
    Vector prefill;
    size_t text_index = 0, hidden_index = 0;
    for (bool mm : multimodal) append(prefill, mm ? row(hidden_projection, hidden_index++) : row(text_projection, text_index++));
    Vector assistant;
    for (int id : {151644, 77091, 198, tokens.front()}) append(assistant, embedding.row(id));
    assistant = project_text(talker_model, assistant, threads);
    Vector trailing_batch;
    size_t trailing_start = 0;
    auto trailing_row = [&](size_t index) {
        // The first text token is consumed in the assistant prefill.
        if (trailing_batch.empty() || index < trailing_start || index >= trailing_start + trailing_batch.size() / 1024) {
            Vector input;
            trailing_start = index;
            for (size_t t = index + 1; t < std::min(tokens.size(), index + 65); ++t) append(input, embedding.row(tokens[t]));
            trailing_batch = project_text(talker_model, input, threads);
        }
        return row(trailing_batch, index - trailing_start);
    };
    Vector special;
    for (int id : {151672, 151673, 151671}) append(special, embedding.row(id));
    special = project_text(talker_model, special, threads);
    const auto bos = row(special, 0), eos_text = row(special, 1), pad = row(special, 2);
    max_frames = std::min(max_frames, 65536 - int(prefill.size() / 1024) - 9);
    if (max_frames < 1) throw std::runtime_error("Talker context limit exceeded");
    auto talker = std::make_unique<Talker>(talker_model, int(prefill.size() / 1024) + 9 + max_frames, threads);
    auto predictor = std::make_unique<Predictor>((models / "code-predictor-staging-bf16.gguf").string(), threads);
    for (int i = 0; i < 3; ++i) append(prefill, row(assistant, i));
    const int prefix_ids[] = {2155, 2156, 2157, speaker, 2148, 2149};
    for (int i = 0; i < 6; ++i) {
        auto value = i < 4 ? pad : i == 4 ? bos : row(assistant, 3);
        const auto codec = talker->embedding(prefix_ids[i]);
        for (int j = 0; j < 1024; ++j) value[j] += codec[j];
        append(prefill, value);
    }

    Vector hidden;
    std::fprintf(stderr, "Omni speech: prefill %zu rows\n", prefill.size() / 1024);
    for (size_t i = 0; i < prefill.size() / 1024; ++i) {
        if (stop()) throw std::runtime_error("voice_cancelled");
        hidden = talker->step(row(prefill, i));
    }
    std::vector<Codes> frames, pending;
    Vector pcm;
    std::bitset<2048> seen_codes;
    int frame_count = 0;
    CodecStream decoder((models / "code2wav-staging-bf16.gguf").string());
    auto flush = [&] {
        if (pending.empty()) return;
        auto samples = decoder.push(pending); pending.clear();
        if (stop()) throw std::runtime_error("voice_cancelled");
        if (emit_audio) emit_audio(samples);
        else pcm.insert(pcm.end(), samples.begin(), samples.end());
    };
    std::mt19937 rng(seed ? *seed : std::random_device{}());
    bool eos = false;
    for (int i = 0; i < max_frames; ++i) {
        if (stop()) throw std::runtime_error("voice_cancelled");
        auto logits = talker->logits(hidden);
        // Match the reference generation repetition penalty on codec group 0.
        // Text token IDs are intentionally excluded from this codec vocabulary.
        for (int previous = 0; previous < 2048; ++previous) {
            if (seen_codes[previous])
                logits[previous] = logits[previous] < 0 ? logits[previous] * 1.05f : logits[previous] / 1.05f;
        }
        const int code = sample_codec(logits, rng, .9f, 1.f, true);
        if (code == 2150) {
            if (size_t(i) < tokens.size()) throw std::runtime_error("voice_audio_incomplete");
            eos = true; break;
        }
        auto prediction = predictor->frame(hidden, code, talker->embedding(code), &rng);
        seen_codes.set(code); ++frame_count;
        if (!use_device_backend()) frames.push_back(prediction.codes);
        if (use_device_backend()) {
            pending.push_back(prediction.codes);
            if (pending.size() == 16) flush();
        }
        if (frame_count % 25 == 0) std::fprintf(stderr, "Omni speech: %d codec frames\n", frame_count);
        const auto text = size_t(i + 1) < tokens.size() ? trailing_row(i) : size_t(i + 1) == tokens.size() ? eos_text : pad;
        for (int j = 0; j < 1024; ++j) prediction.feedback[j] += text[j];
        hidden = talker->step(prediction.feedback);
    }
    if (!frame_count) throw std::runtime_error("Talker produced no audio");
    if (stop()) throw std::runtime_error("voice_cancelled");
    predictor.reset();
    talker.reset();
    if (use_device_backend() && device_available_bytes() < 1024ULL * 1024 * 1024 + device_reserve_bytes()) release_device_models();
    if (!eos) throw std::runtime_error("voice_audio_incomplete");
    if (use_device_backend()) flush();
    else pcm = decode_audio((models / "code2wav-staging-bf16.gguf").string(), frames, threads);
    // A long decoder graph must not force the next Talker segment into paging.
    if (use_device_backend()) release_unused_device_workspaces();
    return {std::move(pcm), frame_count, eos};
}
VoiceResult speak(const std::string & model, const llama_vocab * vocab, const std::vector<CapturedRow> & captured,
                  const std::string & reply, int speaker, int max_frames, int threads, const std::function<bool()> & stop,
                  const std::function<void()> & speech_ready, std::optional<uint32_t> seed,
                  const std::function<void(const Vector &)> & emit_audio) {
    // Small complete phrases keep long TTS aligned with text. This is a work
    // batch size, not a reply limit. Tokenize before any Thinker unload.
    std::vector<std::vector<llama_token>> parts;
    size_t begin = 0, sentence_end = 0;
    for (size_t i = 0; i <= reply.size(); ++i) {
        const bool end = i == reply.size();
        const bool space = !end && (reply[i] == ' ' || reply[i] == '\n' || reply[i] == '\t');
        if (!end && !space) continue;
        if (i <= begin) { begin = i + 1; continue; }
        const char previous = reply[i - 1];
        if (previous == '.' || previous == '!' || previous == '?' || (!end && reply[i] == '\n')) sentence_end = i;
        const auto candidate = tokenize(vocab, reply.substr(begin, i - begin));
        if (end || (candidate.size() >= 160 && sentence_end > begin) || candidate.size() >= 256) {
            const size_t split = end ? i : sentence_end > begin ? sentence_end : i;
            parts.push_back(tokenize(vocab, reply.substr(begin, split - begin)));
            begin = split;
            while (begin < reply.size() && (reply[begin] == ' ' || reply[begin] == '\n' || reply[begin] == '\t')) ++begin;
            sentence_end = begin; i = begin ? begin - 1 : 0;
        }
    }
    if (parts.empty()) throw std::runtime_error("voice_text_empty");
    speech_ready();
    Vector pcm;
    AudioJoin join([&](const Vector & samples) {
        if (stop()) throw std::runtime_error("voice_cancelled");
        if (emit_audio) emit_audio(samples); else append(pcm, samples);
    });
    int frames = 0;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (stop()) throw std::runtime_error("voice_cancelled");
        std::fprintf(stderr, "Omni speech phrase %zu/%zu: %zu text tokens\n", i + 1, parts.size(), parts[i].size());
        join.next_phrase();
        const auto part = speak_part(model, captured, parts[i], speaker, max_frames, threads, stop,
            seed ? std::optional<uint32_t>(*seed + uint32_t(i)) : std::nullopt,
            [&](const Vector & samples) { join.push(samples); });
        if (!use_device_backend()) join.push(part.pcm);
        frames += part.frames;
    }
    join.finish();
    return {std::move(pcm), frames, true};
}
std::string wav_bytes(const Vector & pcm) {
    if (pcm.empty() || pcm.size() > (UINT32_MAX - 36) / 2) throw std::runtime_error("invalid PCM duration");
    std::string output;
    auto u16 = [&](uint16_t v) { output.push_back(char(v & 255)); output.push_back(char(v >> 8)); };
    auto u32 = [&](uint32_t v) { u16(uint16_t(v)); u16(uint16_t(v >> 16)); };
    output += "RIFF"; u32(uint32_t(36 + 2 * pcm.size())); output += "WAVEfmt "; u32(16);
    u16(1); u16(1); u32(24000); u32(48000); u16(2); u16(16); output += "data"; u32(uint32_t(2 * pcm.size()));
    for (float value : pcm) {
        if (!std::isfinite(value)) throw std::runtime_error("nonfinite PCM");
        const double quantized = std::nearbyint(double(std::max(-1.f, std::min(1.f, value))) * 32768.);
        u16(uint16_t(int16_t(std::max(-32768., std::min(32767., quantized)))));
    }
    return output;
}
}
