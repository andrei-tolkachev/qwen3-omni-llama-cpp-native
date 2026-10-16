#include "backend.h"
// The only adapter to llama.cpp server internals. The model kernels use public GGML APIs.
#include "voice.h"
#include "voice_capture.h"
#include "visual_reply.h"
#include "server_security.h"
#include "duplex.h"
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
// Admission is transferable across HTTP/worker threads. A std::mutex may only
// be unlocked by its acquiring thread; it cannot guard a response lifetime.
struct InferenceGate {
    std::atomic<bool> busy{false};
    struct Lease {
        InferenceGate & gate;
        explicit Lease(InferenceGate & gate): gate(gate) {}
        ~Lease() { gate.busy = false; }
    };
    std::shared_ptr<Lease> acquire() {
        bool expected = false;
        if (!busy.compare_exchange_strong(expected, true)) throw lora_omni::DuplexError(409, "omni_inference_busy");
        try { return std::make_shared<Lease>(*this); }
        catch (...) { busy = false; throw; }
    }
};
lora_omni::VoiceCapture capture;
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
    bool duplex = false;
    std::unique_ptr<server_context> context;
    std::unique_ptr<server_routes> routes;
    std::thread worker;
    explicit ThinkerLane(common_params & params): params(params) {}
    ~ThinkerLane() { unload(); }
    void drain() {
        auto reader = context->get_response_reader();
        server_task task(SERVER_TASK_TYPE_SLOT_GET);
        task.id = reader.get_new_id();
        // FIFO, behind cancellation. The public /slots route posts at the
        // front and is therefore not a safe cancellation barrier.
        reader.post_task(std::move(task));
        reader.next([] { return false; });
    }
    void load() {
        if (context) return;
        if (!duplex) lora_omni::release_device_models();
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
        if (duplex || !lora_omni::use_device_backend()) return false;
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
    std::shared_ptr<InferenceGate::Lease> inference_lock;
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<std::string> events;
    std::atomic<bool> cancelled{false};
    bool finished = false;
    std::thread worker;
    using Emit = std::function<void(const std::string &, const json &)>;
    using Generate = std::function<void(const server_http_req &, const Emit &)>;
    VoiceStream(std::shared_ptr<InferenceGate::Lease> lock, const server_http_req & request, Generate generate): inference_lock(std::move(lock)) {
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
        catch (const lora_omni::DuplexError & error) {
            auto response = std::make_unique<server_http_res>();
            response->status = error.status;
            response->data = json({{"error", {{"message", error.what()}, {"type", "omni_duplex_error"}}}}).dump();
            return response;
        }
        catch (const lora_omni::DuplexJson::exception & error) {
            auto response = std::make_unique<server_http_res>();
            response->status = 400;
            response->data = json({{"error", {{"message", "Invalid duplex request fields"}, {"type", "invalid_request"}}}}).dump();
            return response;
        }
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
        const bool duplex_enabled = std::getenv("OMNI_DUPLEX") && std::string(std::getenv("OMNI_DUPLEX")) == "1";
        ThinkerLane lane(params);
        lane.duplex = duplex_enabled;
        if (duplex_enabled) {
            if (!lora_omni::use_device_backend()) throw std::runtime_error("Duplex requires the native GPU backend");
            const char * mode = std::getenv("OMNI_MEMORY_MODE");
            if (mode && std::string(mode) != "auto" && std::string(mode) != "resident")
                throw std::runtime_error("Duplex requires OMNI_MEMORY_MODE=auto or resident; Thinker cannot be unloaded");
            // Place speech weights first, then let llama's fitter place Thinker
            // against actual remaining VRAM. Preserve weights/precision and KV.
            const auto directory = std::getenv("OMNI_SPEECH_MODELS") ? std::filesystem::path(std::getenv("OMNI_SPEECH_MODELS")) : std::filesystem::path(params.model.path).parent_path() / "omni-experimental";
            lora_omni::device_model(lora_omni::talker_model_path(params.model.path));
            lora_omni::device_model((directory / "code-predictor-staging-bf16.gguf").string());
            lora_omni::device_model((directory / "code2wav-staging-bf16.gguf").string(), true);
            for (auto & reserve : params.fit_params_target) reserve = std::max(reserve, size_t(2048ULL * 1024 * 1024));
            params.n_ubatch = std::min(params.n_ubatch, 128);
            params.sleep_idle_seconds = -1;
            params.n_threads_http = std::max(params.n_threads_http, 4);
            params.ctx_shift = false;
        }
        server_http_context http;
        if (!http.init(params)) return 1;
        InferenceGate inference;
        using Route = server_http_context::handler_t server_routes::*;
        auto forward = [&](Route route) {
            return safe([&, route](const server_http_req & request) -> server_http_res_ptr {
                auto lock = inference.acquire();
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
            std::shared_ptr<InferenceGate::Lease> lock;
            server_http_res_ptr upstream;
            LockedResponse(std::shared_ptr<InferenceGate::Lease> held, server_http_res_ptr source): lock(std::move(held)), upstream(std::move(source)) {
                status = upstream->status; data = upstream->data; content_type = upstream->content_type; headers = upstream->headers;
                if (upstream->next) next = [this](std::string & value) { return upstream->next(value); };
            }
            void on_complete() override { upstream->on_complete(); }
        };
        http.post("/v1/chat/completions", safe([&](const server_http_req & request) -> server_http_res_ptr {
            auto lock = inference.acquire();
            lane.load();
            return std::make_unique<LockedResponse>(std::move(lock), lane.routes->post_chat_completions(request));
        }));
        auto process_voice = [&](const server_http_req & request, const VoiceStream::Emit & emit) -> server_http_res_ptr {
            struct SpeechCacheScope {
                bool keep;
                ~SpeechCacheScope() { if (!keep) lora_omni::release_device_models(); }
            } speech_cache_scope{duplex_enabled};
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
            capture.reset(llama_n_ctx(lane.context->get_llama_context())); capture.active = !observe_only;
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
                if (!duplex_enabled && (!setting || std::string(setting) != "resident")) {
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
            auto lock = inference.acquire();
            if (json::parse(request.body).value("stream", false)) {
                return std::make_unique<VoiceStream>(std::move(lock), request,
                    [&](const server_http_req & owned, const VoiceStream::Emit & emit) { process_voice(owned, emit); });
            }
            return process_voice(request, {});
        }));
        std::mutex duplex_mutex;
        std::shared_ptr<lora_omni::DuplexSession> duplex_session;
        using DJson = lora_omni::DuplexJson;
        auto session_for = [&](const DJson & body) {
            std::lock_guard<std::mutex> lock(duplex_mutex);
            const auto id = body.value("session_id", std::string{});
            if (!duplex_session || duplex_session->id != id) throw lora_omni::DuplexError(404, "duplex_session_not_found");
            return duplex_session;
        };
        auto duplex_body = [](const server_http_req & request) {
            if (request.body.size() > 8 * 1024 * 1024) throw lora_omni::DuplexError(413, "duplex_request_limit");
            try {
                auto body = DJson::parse(request.body);
                if (!body.is_object()) throw lora_omni::DuplexError(400, "duplex body must be an object");
                return body;
            }
            catch (...) { throw lora_omni::DuplexError(400, "invalid duplex JSON"); }
        };
        http.post("/v1/omni/duplex/input", safe([&](const server_http_req & request) -> server_http_res_ptr {
            const auto body = duplex_body(request);
            if (!body.contains("sequence") || !body["sequence"].is_number_unsigned() || !body.contains("final") || !body["final"].is_boolean())
                throw lora_omni::DuplexError(400, "sequence and final are required");
            auto response = std::make_unique<server_http_res>();
            response->data = session_for(body)->append(body.at("sequence").get<uint64_t>(), body.at("content"), body.at("final").get<bool>(),
                body.value("replace", false), body.value("visual", DJson::object()), body.value("history_text", std::string{}), body.value("max_tokens", -1),
                body.value("context", std::string{}), body.value("record_input", true), body.value("spoken_text", std::string{}),
                body.value("history_mode", std::string("messages"))).dump();
            return response;
        }));
        http.post("/v1/omni/duplex/context", safe([&](const server_http_req & request) -> server_http_res_ptr {
            const auto body = duplex_body(request);
            auto response = std::make_unique<server_http_res>();
            response->data = session_for(body)->context(body).dump();
            return response;
        }));
        http.post("/v1/omni/duplex/query", safe([&](const server_http_req & request) -> server_http_res_ptr {
            const auto body = duplex_body(request);
            auto response = std::make_unique<server_http_res>();
            response->data = session_for(body)->query(body.at("query"), request.should_stop).dump();
            return response;
        }));
        http.post("/v1/omni/duplex/interrupt-input", safe([&](const server_http_req & request) -> server_http_res_ptr {
            const auto body = duplex_body(request);
            if (!body.contains("sequence") || !body["sequence"].is_number_integer() || body["sequence"].get<int64_t>() < -1)
                throw lora_omni::DuplexError(400, "invalid input sequence");
            auto response = std::make_unique<server_http_res>();
            response->data = session_for(body)->interrupt_input(body["sequence"].get<int64_t>(), body.at("content"), body.value("record_input", true)).dump();
            return response;
        }));
        http.post("/v1/omni/duplex/cancel", safe([&](const server_http_req & request) -> server_http_res_ptr {
            const auto body = duplex_body(request);
            auto response = std::make_unique<server_http_res>();
            response->data = session_for(body)->cancel(body.at("generation_id").get<std::string>()).dump();
            return response;
        }));
        http.post("/v1/omni/duplex/close", safe([&](const server_http_req & request) -> server_http_res_ptr {
            session_for(duplex_body(request))->close();
            auto response = std::make_unique<server_http_res>(); response->data = "{\"closed\":true}"; return response;
        }));
        http.post("/v1/omni/duplex/ack", safe([&](const server_http_req & request) -> server_http_res_ptr {
            const auto body = duplex_body(request);
            if (!body.contains("played_samples") || !body["played_samples"].is_number_unsigned())
                throw lora_omni::DuplexError(400, "played_samples must be a nonnegative integer");
            auto response = std::make_unique<server_http_res>();
            response->data = session_for(body)->acknowledge(body.at("generation_id").get<std::string>(),
                body.at("played_samples").get<uint64_t>(), body.value("final", true)).dump();
            return response;
        }));
        struct DuplexStream : server_http_res {
            std::shared_ptr<InferenceGate::Lease> lease;
            std::shared_ptr<lora_omni::DuplexSession> session;
            DuplexStream(std::shared_ptr<InferenceGate::Lease> lease, std::shared_ptr<lora_omni::DuplexSession> session,
                         const std::function<bool()> & disconnected): lease(std::move(lease)), session(std::move(session)) {
                content_type = "text/event-stream";
                headers["Cache-Control"] = "no-cache, no-transform"; headers["X-Accel-Buffering"] = "no";
                next = [this, disconnected](std::string & data) {
                    lora_omni::DuplexSession::Event event;
                    if (!this->session->next(event, disconnected)) return false;
                    data = "event: " + event.name + "\ndata: " + event.data.dump() + "\n\n";
                    return true;
                };
            }
            void on_complete() override { session->close(); lease.reset(); }
            ~DuplexStream() override { on_complete(); }
        };
        http.post("/v1/omni/duplex", safe([&](const server_http_req & request) -> server_http_res_ptr {
            if (!duplex_enabled) throw lora_omni::DuplexError(409, "start server with OMNI_DUPLEX=1");
            auto lease = inference.acquire();
            const auto body = duplex_body(request);
            // Server-issued opaque IDs fence delayed input/cancel after reopen.
            const auto id = random_string();
            const bool adaptive = body.value("adaptive_dialogue", false);
            lora_omni::DuplexSession::Limits duplex_limits;
            if (adaptive) duplex_limits.playback_window = 24000 * 6;
            duplex_limits.committed_history = adaptive;
            const int max_tokens = body.value("max_tokens", -1);
            const int speaker = body.value("speaker_id", 2301);
            const int frames = body.value("max_audio_frames", 65536);
            const double temperature = body.value("temperature", double(params.sampling.temp));
            if ((max_tokens != -1 && (max_tokens < 1 || max_tokens > 65536)) || speaker < 2301 || speaker > 2303 || frames < 1 || frames > 65536)
                throw lora_omni::DuplexError(400, "invalid duplex generation limits");
            if (!std::isfinite(temperature) || temperature < 0 || temperature > 2)
                throw lora_omni::DuplexError(400, "invalid duplex temperature");
            std::optional<uint32_t> seed;
            if (body.contains("speech_seed")) {
                if (!body["speech_seed"].is_number_unsigned() || body["speech_seed"].get<uint64_t>() > UINT32_MAX)
                    throw lora_omni::DuplexError(400, "invalid speech_seed");
                seed = body["speech_seed"].get<uint32_t>();
            }
            const auto initial = body.value("messages", DJson::array());
            for (const auto & message : initial) {
                if (!message.is_object() || !message.contains("role") || !message.contains("content"))
                    throw lora_omni::DuplexError(400, "invalid initial history");
            }
            auto session = std::make_shared<lora_omni::DuplexSession>(id, initial,
                [&, max_tokens, temperature, adaptive](const DJson & messages, bool final, const lora_omni::DuplexSession::Stop & stop) {
                    // Drain the preceding cancelled task before reusing capture.
                    lane.drain();
                    if (stop()) throw std::runtime_error("voice_cancelled");
                    capture.reset(llama_n_ctx(lane.context->get_llama_context())); capture.active = final;
                    struct Drain {
                        ThinkerLane & lane;
                        ~Drain() {
                            try { lane.drain(); } catch (...) {}
                            capture.active = false;
                        }
                    } drain{lane};
                    auto clean_messages = messages;
                    const auto visual_options = clean_messages.back().value("_omni_visual", DJson::object());
                    clean_messages.back().erase("_omni_visual");
                    const int reserve = clean_messages.back().value("_omni_max_tokens", -1);
                    clean_messages.back().erase("_omni_max_tokens");
                    const auto spoken_text = clean_messages.back().value("_omni_spoken_text", std::string{});
                    clean_messages.back().erase("_omni_spoken_text");
                    const int final_tokens = reserve < 0 ? max_tokens : (max_tokens < 0 ? reserve : std::min(max_tokens, reserve));
                    DJson query = {{"messages", clean_messages}, {"stream", false}, {"temperature", temperature},
                        {"max_tokens", final ? final_tokens : (adaptive ? 96 : (max_tokens < 0 ? 32 : std::min(max_tokens, 32)))}, {"cache_prompt", !final},
                        {"chat_template_kwargs", {{"enable_thinking", true}}}};
                    if (!spoken_text.empty()) {
                        // Force the validated progress utterance while retaining
                        // normal Thinker hidden-state conditioning for Talker.
                        query["grammar"] = "root ::= " + DJson(spoken_text).dump() + "\n";
                        query["chat_template_kwargs"]["enable_thinking"] = false;
                    }
                    const auto serialized = query.dump();
                    const server_http_req prepared{{}, {}, "/v1/chat/completions", "", serialized, {}, stop};
                    auto result = lane.routes->post_chat_completions(prepared);
                    if (stop()) { result.reset(); throw std::runtime_error("voice_cancelled"); }
                    if (result->status != 200) {
                        // Preview input is provisional. Callers can compact the
                        // grounded history before finalizing this utterance.
                        if (!final && result->status == 400) {
                            const auto error = DJson::parse(result->data, nullptr, false);
                            if (error.is_object() && error.contains("error") && error["error"].is_object() &&
                                error["error"].value("type", std::string{}) == "exceed_context_size_error")
                                return lora_omni::DuplexReply{};
                        }
                        throw std::runtime_error("Duplex Thinker: " + result->data);
                    }
                    const auto data = DJson::parse(result->data);
                    result.reset();
                    // Finished responses have synchronized the worker's capture
                    // writes through its result queue. Cancel paths use Drain.
                    lora_omni::DuplexReply reply;
                    lora_omni::VisualReply visual(visual_options.value("frame_id", std::string{}), {}, visual_options.value("allow_screen_request", false));
                    visual.push(data.at("choices").at(0).at("message").at("content").get<std::string>());
                    visual.finish(); reply.text = visual.reply;
                    if (!spoken_text.empty() && reply.text != spoken_text) throw std::runtime_error("voice_spoken_text_mismatch");
                    reply.pointers = DJson::parse(visual.pointers.dump()); reply.actions = DJson::parse(visual.actions.dump());
                    if (final) {
                        if (data.at("choices").at(0).value("finish_reason", "") != "stop") throw std::runtime_error("voice_text_incomplete");
                        if (reply.text.find_first_not_of(" \t\r\n") == std::string::npos) throw std::runtime_error("voice_text_empty");
                        if (!capture.error.empty()) throw std::runtime_error(capture.error);
                        if (capture.overflow) throw std::runtime_error("voice_conditioning_limit");
                        if (capture.rows.empty()) throw std::runtime_error("Thinker capture unavailable");
                        reply.conditioning = std::move(capture.rows);
                    }
                    return reply;
                },
                [&, speaker, frames, seed, adaptive](const lora_omni::DuplexReply & reply, const lora_omni::DuplexSession::Stop & stop,
                                          const lora_omni::DuplexSession::Audio & emit,
                                          const lora_omni::DuplexSession::Boundary & boundary) {
                    // Thinker remains pinned throughout the session. Speech has
                    // its own backend streams, KV, codec state and sampler.
                    const auto * vocab = llama_model_get_vocab(llama_get_model(lane.context->get_llama_context()));
                    lora_omni::DeviceMemoryPressureScope memory_pressure([] {});
                    lora_omni::speak(params.model.path, vocab, reply.conditioning, reply.text, speaker, frames, 4, stop, [] {}, seed, emit, boundary,
                        adaptive ? 12 : 160, adaptive ? 64 : 256);
                }, duplex_limits,
                [&](const DJson & query, const lora_omni::DuplexSession::Stop & stop) {
                    // ASR, web planning and visual observations share the same
                    // Thinker worker as input previews. Talker keeps its own GPU state.
                    lane.drain(); capture.active = false;
                    struct Drain { ThinkerLane & lane; ~Drain() { try { lane.drain(); } catch (...) {} } } drain{lane};
                    if (!query.contains("messages") || !query["messages"].is_array()) throw lora_omni::DuplexError(400, "duplex_query_messages_required");
                    const int tokens = query.value("max_tokens", 768);
                    if (tokens < 1 || tokens > 4096) throw lora_omni::DuplexError(400, "duplex_query_token_limit");
                    DJson payload = {{"messages", query["messages"]}, {"stream", false}, {"temperature", 0},
                        {"max_tokens", tokens}, {"cache_prompt", false}, {"chat_template_kwargs", {{"enable_thinking", true}}}};
                    const server_http_req prepared{{}, {}, "/v1/chat/completions", "", payload.dump(), {}, stop};
                    if (query.value("operation", std::string("chat")) == "context_tokens") {
                        auto result = lane.routes->post_chat_completions_tok(prepared);
                        if (result->status != 200) throw lora_omni::DuplexError(result->status, "Duplex token count: " + result->data);
                        auto measured = DJson::parse(result->data);
                        measured["context_tokens"] = llama_n_ctx(lane.context->get_llama_context());
                        return measured;
                    }
                    auto result = lane.routes->post_chat_completions(prepared);
                    if (stop()) throw lora_omni::DuplexError(410, "duplex_query_cancelled");
                    if (result->status != 200) throw lora_omni::DuplexError(result->status, "Duplex query: " + result->data);
                    const auto choice = DJson::parse(result->data).at("choices").at(0);
                    return DJson{{"text", choice.at("message").at("content")}, {"finish_reason", choice.value("finish_reason", "")}};
                });
            { std::lock_guard<std::mutex> lock(duplex_mutex); duplex_session = session; }
            session->start();
            return std::make_unique<DuplexStream>(std::move(lease), std::move(session), request.should_stop);
        }));

        lane.load();
        if (!http.start()) return 1;
        http.is_ready = true;
        std::signal(SIGTERM, terminate); std::signal(SIGINT, terminate); std::signal(SIGPIPE, SIG_IGN);
        while (!stopping) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        voice_cancelled = true;
        { std::lock_guard<std::mutex> lock(duplex_mutex); if (duplex_session) duplex_session->close(); }
        http.stop(); if (http.thread.joinable()) http.thread.join();
        lane.unload();
        server_stream_session_manager_stop();
        return 0;
    } catch (const std::exception & error) { std::fprintf(stderr, "Omni server: %s\n", error.what()); return 1; }
}
