#pragma once
#include "omni.h"
#include "llama.h"
#include <functional>
#include <optional>
namespace lora_omni {
struct CapturedRow { int token; Vector hidden; };
struct VoiceResult { Vector pcm; int frames; bool eos; };
VoiceResult speak(const std::string & model, const llama_vocab * vocab, const std::vector<CapturedRow> & user,
                  const std::string & reply, int speaker, int max_frames, int threads, const std::function<bool()> & stop,
                  const std::function<void()> & speech_ready = [] {}, std::optional<uint32_t> seed = {},
                  const std::function<void(const Vector &)> & emit_audio = {});
std::string talker_model_path(const std::string & thinker_model);
std::string wav_bytes(const Vector & pcm);
}
