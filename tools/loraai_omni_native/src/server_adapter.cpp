#include "backend.h"
// The only adapter to llama.cpp server internals. The model kernels use public GGML APIs.
#include "voice.h"
#include "visual_reply.h"
#include "server_security.h"
#include "arg.h"
#include "common.h"
#include "server-context.h"
#include "server-http.h"
#include "server-stream.h"
#include "ggml-backend.h"
#include "base64.hpp"
#include <atomic>
#include <csignal>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <filesystem>
#include <condition_variable>
#include <deque>

namespace {
struct Capture {
    std::atomic<bool> active{false};
    std::vector<lora_omni::CapturedRow> rows;
    std::string error;
    bool role_pending = false, in_user = false, overflow = false;
    void reset() {
        rows.clear(); error.clear(); role_pending = in_user = overflow = false;
    }
    void push(lora_omni::CapturedRow row) {
        if (row.token == 151644) {
            role_pending = true;
            in_user = false;
            return;
        }
        if (role_pending) {
            role_pending = false;
            in_user = row.token == 872;
            if (in_user) {
                // Latest user hidden states already attend to the full dialogue.
                rows.clear(); overflow = false;
                rows.push_back({151644, {}});
            }
        }
        if (!in_user) return;
        if (rows.size() >= 3787) { overflow = true; return; }
        rows.push_back(std::move(row));
    }
};
Capture capture;
std::mutex voice_cancel_mutex;
std::string active_voice_id;
std::atomic<bool> voice_cancelled{false};
thread_local const llama_batch * current_batch = nullptr;
thread_local int captured_offset = 0;
volatile std::sig_atomic_t stopping = 0;
void terminate(int) { stopping = 1; }
bool observe(ggml_tensor * tensor, bool ask, void *) {
    if (std::strcmp(tensor->name, "l_out-23") != 0) return false;
    if (ask) return true;
    if (!capture.active) return true;
    try {
        if (!current_batch) throw std::runtime_error("llama_decode adapter was bypassed");
        const int count = int(tensor->ne[1]);
        if (tensor->type != GGML_TYPE_F32 || tensor->ne[0] != 2048 || !ggml_is_contiguous(tensor)
            || captured_offset + count > current_batch->n_tokens)
            throw std::runtime_error("incompatible Thinker layer24 tensor");
        lora_omni::Vector values;
        if (!current_batch->token) {
            values.resize(size_t(count) * 2048);
            ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
        }
        for (int i = 0; i < count; ++i) {
            lora_omni::CapturedRow row;
            row.token = current_batch->token ? current_batch->token[captured_offset + i] : -1;
            if (row.token < 0) row.hidden.assign(values.begin() + size_t(i) * 2048, values.begin() + size_t(i + 1) * 2048);
            capture.push(std::move(row));
        }
        captured_offset += count;
    } catch (const std::exception & error) { capture.error = error.what(); }
    return true;
}
struct ThinkerLane {
    common_params & params;
    std::unique_ptr<server_context> context;
    std::unique_ptr<server_routes> routes;
    std::thread worker;
    explicit ThinkerLane(common_params & params): params(params) {}
    ~ThinkerLane() { unload(); }
    void load() {
        if (context) return;
        lora_omni::release_device_models();
        context = std::make_unique<server_context>();
        routes = std::make_unique<server_routes>(params, *context);
        if (!context->load_model(params)) {
            routes.reset(); context.reset();
            throw std::runtime_error("Thinker model loading failed");
        }
        routes->update_meta(*context);
        worker = std::thread([this] { context->start_loop(); });
    }
    void unload() {
        if (!context) return;
        context->terminate();
        if (worker.joinable()) worker.join();
        routes.reset(); context.reset();
    }
    bool prepare_speech() {
        if (!lora_omni::use_device_backend()) return false;
        const char * setting = std::getenv("OMNI_MEMORY_MODE");
        const std::string mode = setting ? setting : "auto";
        if (mode != "auto" && mode != "staged" && mode != "resident") throw std::runtime_error("OMNI_MEMORY_MODE must be auto, staged or resident");
        auto directory = std::getenv("OMNI_SPEECH_MODELS") ? std::filesystem::path(std::getenv("OMNI_SPEECH_MODELS")) : std::filesystem::path(params.model.path).parent_path() / "omni-experimental";
        const uint64_t weights = std::filesystem::file_size(lora_omni::talker_model_path(params.model.path))
            + std::filesystem::file_size(directory / "code-predictor-staging-bf16.gguf")
            + 2 * std::filesystem::file_size(directory / "code2wav-staging-bf16.gguf");
        const size_t available = lora_omni::device_available_bytes();
        const bool staged = mode == "staged" || (mode == "auto" && available < weights + lora_omni::device_reserve_bytes());
        if (staged) {
            std::fprintf(stderr, "Omni memory: releasing Thinker before GPU speech (%zu bytes available)\n", available);
            unload();
        }
        return staged;
    }
};
struct VoiceStream : server_http_res {
    std::unique_lock<std::mutex> inference_lock;
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<std::string> events;
    std::atomic<bool> cancelled{false};
    bool finished = false;
    std::thread worker;
    using Emit = std::function<void(const std::string &, const json &)>;
    using Generate = std::function<void(const server_http_req &, const Emit &)>;
    VoiceStream(std::unique_lock<std::mutex> lock, const server_http_req & request, Generate generate): inference_lock(std::move(lock)) {
        content_type = "text/event-stream";
        headers["Cache-Control"] = "no-cache, no-transform";
        headers["X-Accel-Buffering"] = "no";
        // Own every request field used by the generator, including the stop callback.
        worker = std::thread([this, params = request.params, headers = request.headers, path = request.path,
                              query = request.query_string, body = request.body, files = request.files,
                              disconnected = request.should_stop, generate] {
            const std::function<bool()> stop = [this, disconnected] { return cancelled.load() || voice_cancelled.load() || disconnected(); };
            const server_http_req owned{params, headers, path, query, body, files, stop};
            const Emit emit = [this, &stop](const std::string & event, const json & value) {
                std::unique_lock<std::mutex> guard(mutex);
                while (events.size() >= 64 && !stop()) changed.wait_for(guard, std::chrono::milliseconds(50));
                if (stop()) throw std::runtime_error("voice_cancelled");
                events.push_back("event: " + event + "\ndata: " + value.dump() + "\n\n");
                changed.notify_all();
            };
            try { generate(owned, emit); }
            catch (const std::exception & error) {
                std::fprintf(stderr, "Omni voice failed: %s\n", error.what());
                std::lock_guard<std::mutex> guard(mutex);
                events.push_back("event: error\ndata: " + json({{"error", error.what()}}).dump() + "\n\n");
            }
            { std::lock_guard<std::mutex> guard(mutex); finished = true; }
            changed.notify_all();
        });
        next = [this, disconnected = request.should_stop](std::string & chunk) {
            std::unique_lock<std::mutex> guard(mutex);
            while (events.empty() && !finished && !cancelled && !disconnected()) changed.wait_for(guard, std::chrono::milliseconds(50));
            if (disconnected() || cancelled) { cancelled = true; changed.notify_all(); return false; }
            if (events.empty()) return false;
            chunk = std::move(events.front()); events.pop_front(); changed.notify_all(); return true;
        };
    }
    void on_complete() override { cancelled = true; changed.notify_all(); if (worker.joinable()) worker.join(); }
    ~VoiceStream() override { on_complete(); }
};
server_http_context::handler_t safe(server_http_context::handler_t handler) {
    return [handler](const server_http_req & request) -> server_http_res_ptr {
        try { return handler(request); }
        catch (const std::exception & error) {
            auto response = std::make_unique<server_http_res>();
            response->status = 500;
            response->data = json({{"error", {{"message", error.what()}, {"type", "omni_error"}}}}).dump();
            return response;
        }
    };
}
}

// Interpose the public C entry point to associate graph callbacks with input rows.
// This also covers mtmd audio/image batches, without modifying upstream source.
extern "C" int32_t llama_decode(llama_context * context, llama_batch batch) {
    using Decode = int32_t (*)(llama_context *, llama_batch);
    static const auto decode = reinterpret_cast<Decode>(dlsym(RTLD_NEXT, "llama_decode"));
    if (!decode) return -1;
    const auto previous = current_batch;
    const auto previous_offset = captured_offset;
    current_batch = &batch;
    captured_offset = 0;
    const int result = decode(context, batch);
    current_batch = previous;
    captured_offset = previous_offset;
    return result;
}

int main(int argc, char ** argv) {
    try {
        lora_omni::DeviceCacheScope device_cache_scope;
        common_init();
        common_params params;
        if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) return 1;
        lora_omni::validate_server_binding(params.hostname, params.api_keys);
        params.n_parallel = 1;
        params.cb_eval = observe;
        llama_backend_init();
        llama_numa_init(params.numa);
        server_stream_session_manager_start();
        ThinkerLane lane(params);
        server_http_context http;
        if (!http.init(params)) return 1;
        std::mutex inference;
        using Route = server_http_context::handler_t server_routes::*;
        auto forward = [&](Route route) {
            return safe([&, route](const server_http_req & request) -> server_http_res_ptr {
                std::unique_lock<std::mutex> lock(inference);
                lane.load();
                return (lane.routes.get()->*route)(request);
            });
        };
        http.get("/health", safe([](const server_http_req &) -> server_http_res_ptr {
            auto response = std::make_unique<server_http_res>(); response->data = "{\"status\":\"ok\"}"; return response;
        }));
        http.get("/v1/models", forward(&server_routes::get_models));
        http.get("/models", forward(&server_routes::get_models));
        http.get("/props", forward(&server_routes::get_props));
        http.get("/slots", forward(&server_routes::get_slots));
        http.post("/slots/:id_slot", forward(&server_routes::post_slots));
        http.post("/tokenize", forward(&server_routes::post_tokenize));
        http.post("/detokenize", forward(&server_routes::post_detokenize));
        http.post("/apply-template", forward(&server_routes::post_apply_template));
        http.post("/v1/chat/completions/input_tokens", forward(&server_routes::post_chat_completions_tok));
        http.post("/chat/completions/input_tokens", forward(&server_routes::post_chat_completions_tok));
        http.post("/v1/omni/cancel", safe([](const server_http_req & request) -> server_http_res_ptr {
            const auto id = json::parse(request.body).value("request_id", std::string{});
            std::lock_guard<std::mutex> guard(voice_cancel_mutex);
            if (!id.empty() && active_voice_id == id) voice_cancelled = true;
            auto response = std::make_unique<server_http_res>(); response->data = "{\"ok\":true}";
            return response;
        }));
        // Normal requests keep the lock through the response stream lifetime.
        struct LockedResponse : server_http_res {
            std::unique_lock<std::mutex> lock;
            server_http_res_ptr upstream;
            LockedResponse(std::unique_lock<std::mutex> held, server_http_res_ptr source): lock(std::move(held)), upstream(std::move(source)) {
                status = upstream->status; data = upstream->data; content_type = upstream->content_type; headers = upstream->headers;
                if (upstream->next) next = [this](std::string & value) { return upstream->next(value); };
            }
            void on_complete() override { upstream->on_complete(); }
        };
        http.post("/v1/chat/completions", safe([&](const server_http_req & request) -> server_http_res_ptr {
            std::unique_lock<std::mutex> lock(inference);
            lane.load();
            return std::make_unique<LockedResponse>(std::move(lock), lane.routes->post_chat_completions(request));
        }));
        auto process_voice = [&](const server_http_req & request, const VoiceStream::Emit & emit) -> server_http_res_ptr {
            lora_omni::DeviceCacheScope speech_cache_scope;
            auto body = json::parse(request.body);
            const bool observe_only = body.value("observe_only", false);
            body.erase("observe_only");
            {
                std::lock_guard<std::mutex> guard(voice_cancel_mutex);
                active_voice_id = body.value("request_id", std::string{});
                voice_cancelled = false;
            }
            struct ClearVoice { ~ClearVoice() { std::lock_guard<std::mutex> guard(voice_cancel_mutex); active_voice_id.clear(); } } clear_voice;
            std::string generation_id = body.value("generation_id", active_voice_id);
            if (generation_id.empty()) generation_id = active_voice_id;
            body.erase("generation_id");
            const bool pcm_only = bool(emit) && body.value("audio_protocol", 1) == 2;
            body.erase("audio_protocol");
            if (emit) emit("start", {{"request_id", active_voice_id}, {"generation_id", generation_id}});
            lane.load();
            body.erase("request_id");
            const int speaker = body.value("speaker_id", 2301);
            const int max_frames = body.value("max_audio_frames", 65536);
            std::optional<uint32_t> speech_seed;
            if (body.contains("speech_seed")) {
                if (!body["speech_seed"].is_number_integer() || body["speech_seed"].get<int64_t>() < 0 ||
                    body["speech_seed"].get<uint64_t>() > UINT32_MAX) throw std::runtime_error("invalid speech_seed");
                speech_seed = body["speech_seed"].get<uint32_t>();
            }
            body.erase("speech_seed");
            const auto frame_id = body.value("visual_frame_id", std::string{});
            body.erase("visual_frame_id");
            const bool allow_screen_request = !observe_only && body.value("allow_screen_request", false);
            body.erase("allow_screen_request");
            if (allow_screen_request && frame_id.empty()) {
                body["grammar"] = R"GBNF(root ::= text "<lora_pointer>" ("null" | "{\"request_screen\":true}") "</lora_pointer>"
text ::= [^<`{}]+
)GBNF";
            }
            if (!frame_id.empty()) {
                // Keep visual commands in a bounded structured suffix while
                // ordinary text still streams token by token.
                body["grammar"] = R"GBNF(root ::= text "<lora_pointer>" ("null" | point) "</lora_pointer>"
text ::= [^<`{}]+
point ::= "{\"x\":" ws number ",\"y\":" ws number ",\"label\":" ws label "}"
number ::= "0" ("." [0-9]{1,3})? | "1" ("." "0"{1,3})?
label ::= "\"" [^"\\\n\r]{0,80} "\""
ws ::= [ \t]*
)GBNF";
            }
            body.erase("speaker_id"); body.erase("max_audio_frames");
            // Qwen3-Omni Instruct uses a plain assistant prefix. The generic
            // Qwen3 no-thinking prefill makes multi-turn audio emit EOS/user roles.
            body["chat_template_kwargs"]["enable_thinking"] = true;
            body["stream"] = bool(emit);
            body["cache_prompt"] = false;
            body["max_tokens"] = body.value("max_tokens", -1);
            const std::string serialized = body.dump();
            const std::function<bool()> stop = [&] { return voice_cancelled.load() || request.should_stop(); };
            const server_http_req prepared{request.params, request.headers, request.path, request.query_string, serialized, request.files, stop};
            capture.reset(); capture.active = !observe_only;
            struct Finish { ~Finish() { capture.active = false; } } finish;
            auto response = lane.routes->post_chat_completions(prepared);
            if (response->status != 200) {
                if (emit) throw std::runtime_error("Thinker generation failed: " + response->data);
                return response;
            }
            lora_omni::VisualReply visual(frame_id, emit, allow_screen_request);
            std::string text;
            std::string finish_reason = "stop";
            if (emit) {
                if (!response->next) throw std::runtime_error("Thinker did not provide a native stream");
                std::string chunk, pending;
                while (!stop() && response->next(chunk)) {
                    pending += chunk;
                    size_t separator;
                    while ((separator = pending.find("\n\n")) != std::string::npos) {
                        const auto record = pending.substr(0, separator); pending.erase(0, separator + 2);
                        const auto data_at = record.find("data: ");
                        if (data_at == std::string::npos) continue;
                        const auto data = record.substr(data_at + 6);
                        if (data == "[DONE]") continue;
                        const auto item = json::parse(data);
                        if (item.contains("error")) throw std::runtime_error(item.at("error").dump());
                        if (!item.contains("choices") || item["choices"].empty()) continue;
                        if (item["choices"][0].value("finish_reason", json()).is_string())
                            finish_reason = item["choices"][0]["finish_reason"].get<std::string>();
                        const auto delta = item["choices"][0].value("delta", json::object());
                        if (delta.contains("content") && delta["content"].is_string()) {
                            const auto part = delta["content"].get<std::string>();
                            if (!part.empty()) visual.push(part);
                        }
                    }
                }
                response->on_complete();
                response = std::make_unique<server_http_res>();
                if (stop()) throw std::runtime_error("voice_cancelled");
            } else {
                const auto result = json::parse(response->data);
                visual.push(result.at("choices").at(0).at("message").at("content").get<std::string>());
                finish_reason = result.at("choices").at(0).value("finish_reason", std::string("stop"));
                // Release the upstream task reader before staged mode frees
                // its context; the outer response owns only plain JSON data.
                response->on_complete();
                response = std::make_unique<server_http_res>();
            }
            visual.finish(); text = visual.reply;
            capture.active = false;
            if (stop()) throw std::runtime_error("voice_cancelled");
            if (observe_only) {
                response->data = json({{"text", text}, {"finish_reason", finish_reason}, {"observation", true}}).dump();
                if (emit) emit("done", json::parse(response->data));
                return response;
            }
            if (text.find_first_not_of(" \t\r\n") == std::string::npos) throw std::runtime_error("voice_text_empty");
            if (finish_reason != "stop") throw std::runtime_error("voice_text_incomplete");
            if (!capture.error.empty()) throw std::runtime_error(capture.error);
            if (capture.overflow) throw std::runtime_error("voice_conditioning_limit");
            if (capture.rows.empty()) throw std::runtime_error("Thinker capture unavailable; incompatible llama.cpp adapter");
            if (emit) emit("phase", {{"phase", "speech"}});
            const auto * vocab = llama_model_get_vocab(llama_get_model(lane.context->get_llama_context()));
            bool staged = false;
            lora_omni::DeviceMemoryPressureScope memory_pressure([&] {
                const char * setting = std::getenv("OMNI_MEMORY_MODE");
                if (!setting || std::string(setting) != "resident") {
                    std::fprintf(stderr, "Omni memory: releasing Thinker for measured GPU decode workspace\n");
                    lane.unload(); staged = true;
                }
            });
            size_t pcm_offset = 0, pcm_sequence = 0;
            auto voice = lora_omni::speak(params.model.path, vocab, capture.rows, text, speaker, max_frames, 4,
                [&]() { return voice_cancelled.load() || request.should_stop(); },
                [&]() { staged = lane.prepare_speech(); }, speech_seed,
                pcm_only ? std::function<void(const lora_omni::Vector &)>([&](const lora_omni::Vector & samples) {
                    if (emit) {
                        const auto bytes = lora_omni::wav_bytes(samples).substr(44);
                        emit("audio_chunk", {{"generation_id", generation_id}, {"sequence", pcm_sequence++}, {"sample_offset", pcm_offset},
                            {"format", "pcm_s16le"}, {"sample_rate", 24000}, {"channels", 1},
                            {"data", base64::encode(bytes)}});
                    }
                    pcm_offset += samples.size();
                }) : std::function<void(const lora_omni::Vector &)>{});
            json audio = {{"format", "pcm_s16le"}, {"sample_rate", 24000}, {"channels", 1}, {"samples", pcm_offset}, {"chunks", pcm_sequence}};
            if (!pcm_only) audio = {{"format", "wav"}, {"sample_rate", 24000}, {"data", base64::encode(lora_omni::wav_bytes(voice.pcm))}};
            response->data = json({{"generation_id", generation_id}, {"text", text}, {"pointers", visual.pointers}, {"actions", visual.actions}, {"finish_reason", finish_reason}, {"audio", audio},
                {"audio_frames", voice.frames}, {"audio_finish_reason", voice.eos ? "stop" : "length"},
                {"thinker_rows", capture.rows.size()},
                {"multimodal_rows", std::count_if(capture.rows.begin(), capture.rows.end(), [](const auto & row) { return row.token < 0; })},
                {"model", "lora_omni"}, {"speech_backend", lora_omni::use_device_backend() ? "gpu" : "cpu"}, {"memory_mode", staged ? "staged" : "resident"}}).dump();
            capture.rows.clear();
            if (emit) {
                const auto output = json::parse(response->data);
                emit("done", output);
            }
            return response;
        };
        http.post("/v1/omni/voice", safe([&](const server_http_req & request) -> server_http_res_ptr {
            std::unique_lock<std::mutex> lock(inference);
            if (json::parse(request.body).value("stream", false)) {
                return std::make_unique<VoiceStream>(std::move(lock), request,
                    [&](const server_http_req & owned, const VoiceStream::Emit & emit) { process_voice(owned, emit); });
            }
            return process_voice(request, {});
        }));
        lane.load();
        if (!http.start()) return 1;
        http.is_ready = true;
        std::signal(SIGTERM, terminate); std::signal(SIGINT, terminate); std::signal(SIGPIPE, SIG_IGN);
        while (!stopping) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        voice_cancelled = true;
        http.stop(); if (http.thread.joinable()) http.thread.join();
        lane.unload();
        server_stream_session_manager_stop();
        return 0;
    } catch (const std::exception & error) { std::fprintf(stderr, "Omni server: %s\n", error.what()); return 1; }
}
