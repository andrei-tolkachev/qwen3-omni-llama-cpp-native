#include "duplex.h"
#include "base64.hpp"
#include <chrono>

namespace lora_omni {
namespace {
struct InputAudio {
    uint32_t rate = 0;
    uint16_t channels = 0;
    std::string pcm;
};
uint32_t le(const std::string & bytes, size_t at, size_t n) {
    if (at > bytes.size() || n > bytes.size() - at) throw DuplexError(400, "duplex_invalid_wav");
    uint32_t value = 0;
    for (size_t i = 0; i < n; ++i) value |= uint32_t(uint8_t(bytes[at + i])) << (8 * i);
    return value;
}
InputAudio decode_input_audio(const DuplexJson & block) {
    try {
        const auto & audio = block.at("input_audio");
        const auto encoded = audio.at("data").get<std::string>();
        const auto raw = base64::decode(encoded);
        if (base64::encode(raw) != encoded) throw DuplexError(400, "duplex_invalid_base64");
        InputAudio result;
        const auto format = audio.at("format").get<std::string>();
        if (format == "pcm16") {
            const auto rate = audio.at("sample_rate").get<int64_t>();
            const auto channels = audio.value("channels", int64_t(1));
            if (rate < 8000 || rate > 96000 || channels < 1 || channels > 2)
                throw DuplexError(400, "duplex_invalid_pcm_format");
            result.rate = uint32_t(rate); result.channels = uint16_t(channels); result.pcm = raw;
        } else if (format == "wav") {
            if (raw.size() < 12 || raw.substr(0, 4) != "RIFF" || raw.substr(8, 4) != "WAVE" || le(raw, 4, 4) != raw.size() - 8)
                throw DuplexError(400, "duplex_invalid_wav");
            bool fmt = false, data = false;
            for (size_t at = 12; at < raw.size();) {
                const auto size = le(raw, at + 4, 4);
                if (size > raw.size() - at - 8) throw DuplexError(400, "duplex_invalid_wav");
                const auto name = raw.substr(at, 4);
                if (name == "fmt ") {
                    if (fmt || size < 16 || le(raw, at + 8, 2) != 1 || le(raw, at + 22, 2) != 16)
                        throw DuplexError(400, "duplex_requires_s16le_wav");
                    result.channels = uint16_t(le(raw, at + 10, 2)); result.rate = le(raw, at + 12, 4); fmt = true;
                    if (result.channels < 1 || result.channels > 2 || result.rate < 8000 || result.rate > 96000 ||
                        le(raw, at + 20, 2) != result.channels * 2 || le(raw, at + 16, 4) != result.rate * result.channels * 2)
                        throw DuplexError(400, "duplex_invalid_wav_format");
                } else if (name == "data") {
                    if (data) throw DuplexError(400, "duplex_duplicate_wav_data");
                    result.pcm = raw.substr(at + 8, size); data = true;
                }
                at += 8 + size + (size & 1);
                if (at > raw.size()) throw DuplexError(400, "duplex_invalid_wav_padding");
            }
            if (!fmt || !data) throw DuplexError(400, "duplex_incomplete_wav");
        } else throw DuplexError(400, "duplex_audio_format_must_be_wav_or_pcm16");
        if (result.pcm.empty() || result.pcm.size() % (2 * result.channels)) throw DuplexError(400, "duplex_unaligned_pcm");
        return result;
    } catch (const DuplexError &) { throw; }
    catch (...) { throw DuplexError(400, "duplex_invalid_input_audio"); }
}
DuplexJson encode_input_audio(const InputAudio & audio) {
    std::string wav;
    auto put = [&](uint32_t value, int n) { for (int i = 0; i < n; ++i) wav.push_back(char((value >> (i * 8)) & 255)); };
    wav += "RIFF"; put(uint32_t(audio.pcm.size() + 36), 4); wav += "WAVEfmt "; put(16, 4);
    put(1, 2); put(audio.channels, 2); put(audio.rate, 4); put(audio.rate * audio.channels * 2, 4);
    put(audio.channels * 2, 2); put(16, 2); wav += "data"; put(uint32_t(audio.pcm.size()), 4); wav += audio.pcm;
    return {{"type", "input_audio"}, {"input_audio", {{"format", "wav"}, {"data", base64::encode(wav)}}}};
}
void append_content(DuplexJson & target, const DuplexJson & content) {
    for (const auto & block : content) {
        if (block.at("type") == "input_audio") {
            auto audio = decode_input_audio(block);
            if (!target.empty() && target.back().at("type") == "input_audio") {
                auto previous = decode_input_audio(target.back());
                if (previous.rate != audio.rate || previous.channels != audio.channels)
                    throw DuplexError(400, "duplex_audio_format_changed_within_turn");
                previous.pcm += audio.pcm;
                target.back() = encode_input_audio(previous);
            } else target.push_back(encode_input_audio(audio));
        } else if (block.at("type") == "text") {
            if (!block.contains("text") || !block["text"].is_string()) throw DuplexError(400, "duplex_invalid_text");
            if (!target.empty() && target.back().at("type") == "text")
                target.back()["text"] = target.back()["text"].get<std::string>() + block["text"].get<std::string>();
            else target.push_back(block);
        } else target.push_back(block);
    }
}
}
DuplexSession::DuplexSession(std::string id, DuplexJson history, Think think, Speak speak, Limits limits, Query query):
    id(std::move(id)), think(std::move(think)), speak(std::move(speak)), query_fn(std::move(query)), limits(limits), history(std::move(history)) {
    if (!this->history.is_array() || this->history.dump().size() > limits.history_bytes)
        throw DuplexError(413, "duplex_history_limit");
}
DuplexSession::~DuplexSession() { close(); }
void DuplexSession::start() {
    emit("session", {{"session_id", id}, {"protocol", 1}, {"next_sequence", 0}, {"revision", 0}, {"playback_ack_required", true},
        {"capabilities", {"query", "replace_input", "visual", "context", "interrupt_input", "playback_final", "input_context", "adaptive_dialogue", "committed_prefix", "verbatim_speech"}}});
    try {
        thinker = std::thread([this] { think_loop(); });
        talker = std::thread([this] { speech_loop(); });
    } catch (...) { close(); throw; }
}
DuplexJson DuplexSession::append(uint64_t sequence, const DuplexJson & content, bool final, bool replace, const DuplexJson & visual, const std::string & history_text, int max_tokens, const std::string & context, bool record_input, const std::string & spoken_text) {
    if (!spoken_text.empty() && (!final || record_input || spoken_text.size() > 1024 || spoken_text.find_first_of("\r\n\t") != std::string::npos))
        throw DuplexError(400, "duplex_invalid_spoken_text");
    if (max_tokens != -1 && (max_tokens < 1 || max_tokens > 65536)) throw DuplexError(400, "duplex_token_limit");
    if (history_text.size() > 65536) throw DuplexError(413, "duplex_history_text_limit");
    if (!visual.is_object() || visual.size() > 2) throw DuplexError(400, "duplex_invalid_visual_metadata");
    for (const auto & item : visual.items()) {
        if (item.key() == "frame_id") {
            if (!item.value().is_string()) throw DuplexError(400, "duplex_invalid_frame_id");
            const auto id = item.value().get<std::string>();
            if (id.size() != 32 || id.find_first_not_of("0123456789abcdef") != std::string::npos)
                throw DuplexError(400, "duplex_invalid_frame_id");
        } else if (item.key() != "allow_screen_request" || !item.value().is_boolean())
            throw DuplexError(400, "duplex_invalid_visual_metadata");
    }
    if (!content.is_array()) throw DuplexError(400, "content must be an array of text/input_audio/image_url blocks");
    for (const auto & block : content) {
        if (!block.is_object() || !block.contains("type") || !block["type"].is_string())
            throw DuplexError(400, "invalid duplex content block");
        const auto type = block["type"].get<std::string>();
        if (type != "text" && type != "input_audio" && type != "image_url")
            throw DuplexError(400, "unsupported duplex content block");
    }
    if (context.size() > limits.input_bytes) throw DuplexError(413, "duplex_input_limit");
    const auto bytes = content.dump().size() + context.size() + spoken_text.size();
    std::lock_guard<std::mutex> lock(mutex);
    if (closed) throw DuplexError(410, "duplex_session_closed");
    if (sequence != expected_sequence) throw DuplexError(409, "duplex_sequence_conflict");
    if (input && input->final) throw DuplexError(429, "duplex_input_backpressure");
    const size_t old_bytes = input && !replace ? input->bytes : 0;
    if (bytes > limits.input_bytes - old_bytes || content.size() > limits.fragments)
        throw DuplexError(413, "duplex_input_limit");
    if (content.empty() && (replace || !final || !input || input->content.empty()))
        throw DuplexError(400, "duplex_empty_input");
    // Concatenate adjacent PCM/text fragments losslessly before mtmd. Encoding
    // separate WAV fragments as separate audio messages changes model context.
    auto combined = input && !replace ? input->content : DuplexJson::array();
    append_content(combined, content);
    if (combined.size() > limits.fragments) throw DuplexError(413, "duplex_input_limit");
    if (!input) input = std::make_unique<Input>();
    input->content = std::move(combined);
    input->bytes = old_bytes + bytes;
    if (replace || !visual.empty()) input->visual = visual;
    input->history_text = history_text;
    input->record_input = record_input;
    input->spoken_text = spoken_text;
    if (replace || !context.empty()) input->context = context;
    input->max_tokens = max_tokens;
    input->sequence = sequence; input->version = ++expected_sequence; input->final = final;
    // A growing microphone buffer must not starve an in-flight preview. It
    // finishes its immutable snapshot, then the worker takes the newest one.
    // Finalization/replacement still fences it immediately.
    if (final || replace) ++epoch;
    changed.notify_all();
    return {{"accepted_sequence", sequence}, {"next_sequence", expected_sequence}, {"revision", revision}};
}
DuplexJson DuplexSession::interrupt_input(int64_t sequence, const DuplexJson & content, bool record_input) {
    if (!content.is_array() || content.size() > limits.fragments || content.dump().size() > limits.input_bytes)
        throw DuplexError(413, "duplex_input_limit");
    if (content.empty()) {
        std::lock_guard<std::mutex> lock(mutex);
        if (closed) throw DuplexError(410, "duplex_session_closed");
        if (!input || input->final || int64_t(input->sequence) != sequence)
            throw DuplexError(409, "duplex_input_sequence_conflict");
        input.reset(); ++epoch; ++revision; changed.notify_all();
        return {{"interrupted", true}, {"revision", revision}};
    }
    DuplexJson clean = DuplexJson::array();
    for (const auto & block : content) {
        if (!block.is_object() || !block.contains("type") || !block["type"].is_string()) throw DuplexError(400, "invalid duplex content block");
        const auto type = block["type"].get<std::string>();
        if (type != "text" && type != "input_audio" && type != "image_url") throw DuplexError(400, "unsupported duplex content block");
    }
    append_content(clean, content);
    std::string promoted;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (closed) throw DuplexError(410, "duplex_session_closed");
        if (active && int64_t(active->sequence) == sequence) promoted = active->generation;
        else {
            if ((input && int64_t(input->sequence) != sequence) || (!input && sequence != -1))
                throw DuplexError(409, "duplex_input_sequence_conflict");
            auto updated = history;
            if (record_input) updated.push_back({{"role", "user"}, {"content", clean}});
            if (updated.dump().size() > limits.history_bytes) throw DuplexError(413, "duplex_history_limit");
            history = std::move(updated); input.reset(); ready.reset(); ++epoch; ++revision;
            changed.notify_all();
            return {{"interrupted", true}, {"revision", revision}};
        }
    }
    auto result = cancel(promoted); result["interrupted"] = result["cancelled"];
    return result;
}
DuplexJson DuplexSession::context(const DuplexJson & request) {
    std::lock_guard<std::mutex> lock(mutex);
    if (closed) throw DuplexError(410, "duplex_session_closed");
    size_t tail = std::min(size_t(4), history.size());
    if (active && !active->cancelled) tail = std::max(tail, history.size() - (active->assistant_index ? active->assistant_index - 1 : 0));
    if (request.contains("messages")) {
        const auto & replacement = request.at("messages");
        if (!request.contains("revision") || request["revision"] != revision)
            throw DuplexError(409, "duplex_context_changed");
        if (!replacement.is_array() || replacement.size() < tail || replacement.dump().size() > limits.history_bytes)
            throw DuplexError(413, "duplex_history_limit");
        for (const auto & message : replacement)
            if (!message.is_object() || !message.contains("role") || !message.contains("content"))
                throw DuplexError(400, "duplex_invalid_history");
        for (size_t i = 0; i < tail; ++i)
            if (replacement[replacement.size() - 1 - i] != history[history.size() - 1 - i])
                throw DuplexError(409, "duplex_context_tail_changed");
        if (active && !active->cancelled)
            active->assistant_index = replacement.size() - (history.size() - active->assistant_index);
        history = replacement; ready.reset(); ++revision; ++epoch; changed.notify_all();
    }
    return {{"messages", history}, {"revision", revision}, {"preserve_tail", tail}};
}
DuplexJson DuplexSession::query(const DuplexJson & request, const Stop & disconnected) {
    if (!query_fn) throw DuplexError(501, "duplex_queries_unavailable");
    if (!request.is_object() || request.dump().size() > limits.input_bytes)
        throw DuplexError(413, "duplex_query_limit");
    auto job = std::make_shared<QueryJob>(); job->request = request; job->disconnected = disconnected;
    std::unique_lock<std::mutex> lock(mutex);
    if (closed) throw DuplexError(410, "duplex_session_closed");
    if (pending_query) throw DuplexError(429, "duplex_query_backpressure");
    pending_query = job; ++epoch; changed.notify_all();
    while (!job->done && !closed && !disconnected()) changed.wait_for(lock, std::chrono::milliseconds(50));
    if (!job->done) throw DuplexError(410, "duplex_query_cancelled");
    if (job->error) std::rethrow_exception(job->error);
    return job->result;
}
bool DuplexSession::emit(std::string name, DuplexJson data, const Stop & stop) {
    data["session_id"] = id;
    const auto bytes = name.size() + data.dump().size();
    if (bytes > limits.event_bytes) throw DuplexError(413, "duplex_event_limit");
    std::unique_lock<std::mutex> lock(event_mutex);
    while (!closed && !(stop && stop()) && (events.size() >= limits.events || queued_bytes + bytes > limits.event_bytes))
        event_changed.wait_for(lock, std::chrono::milliseconds(20));
    if (closed || (stop && stop())) return false;
    queued_bytes += bytes;
    events.push_back({std::move(name), std::move(data)});
    event_changed.notify_all();
    return true;
}
bool DuplexSession::next(Event & event, const Stop & disconnected) {
    std::unique_lock<std::mutex> lock(event_mutex);
    if (!event_changed.wait_for(lock, std::chrono::seconds(1), [&] { return !events.empty() || closed || disconnected(); })) {
        event = {"keepalive", {{"session_id", id}}}; return true;
    }
    if (disconnected()) return false;
    if (events.empty()) return false;
    event = std::move(events.front()); events.pop_front();
    queued_bytes -= event.name.size() + event.data.dump().size();
    event_changed.notify_all();
    return true;
}
void DuplexSession::fail(const std::string & error) {
    // A worker failure is terminal; do not block trying to report into a full
    // output queue. Retain a single error, cancel its peer, and drain on close.
    { std::lock_guard<std::mutex> lock(event_mutex);
      { std::lock_guard<std::mutex> state(mutex); closed = true; ++epoch; }
      events.clear(); events.push_back({"error", {{"session_id", id}, {"error", error}}});
      queued_bytes = events.front().name.size() + events.front().data.dump().size(); }
    changed.notify_all(); event_changed.notify_all();
}
void DuplexSession::think_loop() {
    try {
        auto next_partial = std::chrono::steady_clock::time_point::min();
        while (!closed) {
            DuplexJson messages;
            std::shared_ptr<QueryJob> auxiliary;
            uint64_t version, rev, seq, fence;
            bool final;
            {
                std::unique_lock<std::mutex> lock(mutex);
                changed.wait(lock, [&] { return closed || pending_query || (input && !ready &&
                    (input->version != processed_version || revision != processed_revision)); });
                if (closed) break;
                if (!pending_query && !input->final) changed.wait_until(lock, next_partial, [&] { return closed || pending_query || !input || input->final; });
                if (closed) break;
                if (!pending_query && !input) continue;
                if (pending_query) {
                    auxiliary = pending_query; pending_query.reset();
                } else {
                next_partial = std::chrono::steady_clock::now() + limits.partial_interval;
                version = input->version; rev = revision; seq = input->sequence;
                fence = epoch; final = input->final;
                messages = history;
                // Ephemeral grounding belongs to Thinker's context, not to the
                // speaker's utterance. Keep it out of Talker conditioning and
                // committed dialogue. Place turn-local context immediately before
                // this input: earlier assistant replies must not displace the
                // caller's current evidence or continuation instructions.
                if (!input->context.empty()) {
                    messages.push_back({{"role", "system"}, {"content", input->context}});
                }
                messages.push_back({{"role", "user"}, {"content", input->content}, {"_omni_visual", input->visual}, {"_omni_max_tokens", input->max_tokens}, {"_omni_spoken_text", input->spoken_text}});
                }
            }
            if (auxiliary) {
                const Stop cancelled = [this, auxiliary] { return closed || auxiliary->disconnected(); };
                try {
                    if (cancelled()) throw DuplexError(410, "duplex_query_cancelled");
                    auxiliary->result = query_fn(auxiliary->request, cancelled);
                } catch (...) { auxiliary->error = std::current_exception(); }
                { std::lock_guard<std::mutex> lock(mutex); auxiliary->done = true; }
                changed.notify_all();
                continue;
            }
            const Stop stop = [this, fence] { return closed || epoch.load() != fence; };
            emit("input_started", {{"sequence", seq}, {"revision", rev}, {"final", final}}, stop);
            DuplexReply reply;
            const auto started = std::chrono::steady_clock::now();
            try { reply = think(messages, final, stop); }
            catch (...) { if (stop()) continue; throw; }
            if (stop()) continue;
            const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
            const auto preview = reply.text;
            std::string gen;
            std::shared_ptr<Job> prepared;
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (stop()) continue;
                processed_version = version; processed_revision = rev;
                if (final) {
                    prepared = std::make_shared<Job>();
                    prepared->reply = std::move(reply); prepared->revision = rev;
                    prepared->version = version; prepared->sequence = seq;
                    prepared->generation = id + ":" + std::to_string(++generation);
                    gen = prepared->generation;
                }
            }
            emit("input_processed", {{"sequence", seq}, {"revision", rev}, {"final", final},
                {"text", preview}, {"generation_id", gen}, {"thinker_ms", ms}}, stop);
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (prepared && !stop()) ready = std::move(prepared);
            }
            changed.notify_all();
        }
    } catch (const std::exception & e) { if (!closed) fail(e.what()); }
}
void DuplexSession::speech_loop() {
    try {
        while (!closed) {
            std::shared_ptr<Job> job;
            {
                std::unique_lock<std::mutex> lock(mutex);
                changed.wait(lock, [&] { return closed || bool(ready); });
                if (closed) break;
                if (ready->revision != revision) { ready.reset(); changed.notify_all(); continue; }
                job = ready; ready.reset();
                DuplexJson updated = history;
                if (input->record_input) updated.push_back({{"role", "user"}, {"content", input->history_text.empty() ? input->content : DuplexJson(input->history_text)}});
                // The plan stays private to the job. Adaptive dialogue history
                // contains only the prefix confirmed by actual playback.
                updated.push_back({{"role", "assistant"}, {"content", limits.committed_history ? "" : job->reply.text}});
                if (updated.dump().size() > limits.history_bytes) throw DuplexError(413, "duplex_history_limit");
                history = std::move(updated); input.reset();
                job->assistant_index = history.size() - 1;
                active = job; ++revision; ++epoch;
                job->revision = revision;
                changed.notify_all();
            }
            const Stop stop = [this, job] { return closed || job->cancelled.load(); };
            emit("speech_started", {{"generation_id", job->generation}, {"sequence", job->sequence},
                {"revision", job->revision}, {"text", job->reply.text},
                {"pointers", job->reply.pointers}, {"actions", job->reply.actions}}, stop);
            size_t offset = 0, sequence = 0;
            const auto started = std::chrono::steady_clock::now();
            try {
                speak(job->reply, stop, [&](const Vector & samples) {
                    if (stop()) throw std::runtime_error("voice_cancelled");
                    if (limits.playback_window) {
                        std::unique_lock<std::mutex> lock(mutex);
                        changed.wait(lock, [&] { return stop() || offset <= job->played_samples + limits.playback_window; });
                        if (stop()) throw std::runtime_error("voice_cancelled");
                    }
                    const auto bytes = wav_bytes(samples).substr(44);
                    { std::lock_guard<std::mutex> lock(mutex); job->samples = offset + samples.size(); }
                    if (!emit("audio_chunk", {{"generation_id", job->generation}, {"sequence", sequence++},
                        {"sample_offset", offset}, {"sample_rate", 24000}, {"channels", 1},
                        {"format", "pcm_s16le"}, {"data", base64::encode(bytes)}}, stop))
                        throw std::runtime_error("voice_cancelled");
                    offset += samples.size();
                }, [&](size_t text_end, size_t sample_end) {
                    { std::lock_guard<std::mutex> lock(mutex); job->boundaries.emplace_back(text_end, sample_end); commit_played(job); }
                    emit("phrase_boundary", {{"generation_id", job->generation}, {"text_end", text_end},
                        {"sample_end", sample_end}}, stop);
                });
            } catch (...) { if (!stop()) throw; }
            const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
            { std::lock_guard<std::mutex> lock(mutex);
              job->synthesized = true; job->completed = job->playback_final && job->played_samples == job->samples; commit_played(job); }
            if (!stop()) emit("speech_done", {{"generation_id", job->generation}, {"samples", offset},
                {"chunks", sequence}, {"speech_ms", ms}, {"finish_reason", "stop"}}, stop);
            {
                std::unique_lock<std::mutex> lock(mutex);
                // Keep the output generation alive after synthesis until its
                // actual playback is acknowledged. Thinker keeps running.
                changed.wait(lock, [&] { return stop() || job->completed; });
                active.reset();
                changed.notify_all();
            }
        }
    } catch (const std::exception & e) { if (!closed) fail(e.what()); }
}
DuplexJson DuplexSession::cancel(const std::string & gen) {
    uint64_t rev;
    std::string heard_text;
    uint64_t heard_samples = 0, played_samples = 0;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!active || active->generation != gen || active->cancelled || active->completed)
            return {{"cancelled", false}, {"revision", revision}};
        commit_played(active);
        active->cancelled = true;
        // Adaptive history already contains the heard prefix. A cancelled
        // draft must never replace or shorten those committed words.
        size_t heard = 0;
        for (const auto & boundary : active->boundaries)
            if (boundary.second <= active->played_samples) { heard = boundary.first; heard_samples = boundary.second; }
        heard_text = active->reply.text.substr(0, heard); played_samples = active->played_samples;
        if (heard && !limits.committed_history) history[active->assistant_index]["content"] = active->reply.text.substr(0, heard);
        else if (!heard) history.erase(history.begin() + active->assistant_index);
        rev = ++revision; ++epoch; ready.reset();
        changed.notify_all();
    }
    {
        std::lock_guard<std::mutex> lock(event_mutex);
        for (auto it = events.begin(); it != events.end();) {
            if (it->data.value("generation_id", std::string{}) == gen) {
                queued_bytes -= it->name.size() + it->data.dump().size(); it = events.erase(it);
            } else ++it;
        }
        event_changed.notify_all();
    }
    // Control replies carry the fence immediately, even with a stalled reader.
    return {{"cancelled", true}, {"generation_id", gen}, {"revision", rev},
        {"heard_text", heard_text}, {"heard_samples", heard_samples}, {"played_samples", played_samples}};
}
DuplexJson DuplexSession::acknowledge(const std::string & gen, uint64_t samples, bool final) {
    std::lock_guard<std::mutex> lock(mutex);
    if (closed) throw DuplexError(410, "duplex_session_closed");
    if (!active || active->generation != gen || active->cancelled)
        throw DuplexError(409, "duplex_inactive_generation");
    if (samples < active->played_samples || samples > active->samples)
        throw DuplexError(409, "duplex_playback_offset_conflict");
    active->played_samples = samples;
    active->playback_final = final;
    active->completed = final && active->synthesized && samples == active->samples;
    commit_played(active);
    changed.notify_all();
    return {{"generation_id", gen}, {"played_samples", samples}, {"completed", active->completed}, {"revision", revision}};
}
void DuplexSession::commit_played(const std::shared_ptr<Job> & job) {
    if (!limits.committed_history || job->cancelled) return;
    size_t end = job->committed_text;
    for (const auto & boundary : job->boundaries)
        if (boundary.second <= job->played_samples) end = std::max(end, boundary.first);
    if (job->completed) end = job->reply.text.size();
    if (end == job->committed_text) return;
    auto & content = history[job->assistant_index]["content"].get_ref<std::string &>();
    if (end > job->reply.text.size() || content != job->reply.text.substr(0, job->committed_text))
        throw DuplexError(409, "duplex_committed_prefix_changed");
    content.append(job->reply.text, job->committed_text, end-job->committed_text);
    job->committed_text = end;
    ++revision; ++epoch; ready.reset(); changed.notify_all();
}
void DuplexSession::close() {
    // Pair the terminal predicate with its wait mutex, or a close between a
    // predicate check and wait can lose the notification and hang join().
    { std::lock_guard<std::mutex> lock(mutex); closed = true; ++epoch; }
    changed.notify_all(); event_changed.notify_all();
    std::lock_guard<std::mutex> join(join_mutex);
    if (thinker.joinable()) thinker.join();
    if (talker.joinable()) talker.join();
}
}
