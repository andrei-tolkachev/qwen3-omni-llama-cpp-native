#pragma once
#include "voice.h"
#include "json.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

namespace lora_omni {
using DuplexJson = nlohmann::ordered_json;
struct DuplexError : std::runtime_error {
    int status;
    DuplexError(int status, const std::string & message): std::runtime_error(message), status(status) {}
};
struct DuplexReply {
    std::string text;
    std::vector<CapturedRow> conditioning;
    DuplexJson pointers = DuplexJson::array(), actions = DuplexJson::array();
};

// One Thinker owner and one speech owner. Neither calls the other while holding
// the dialogue mutex. GPU contexts, KV and conditioning are never shared mutable
// state. The transport has its own bounded queue and cannot hold the input lock.
class DuplexSession {
public:
    using Stop = std::function<bool()>;
    using Think = std::function<DuplexReply(const DuplexJson &, bool, const Stop &)>;
    using Audio = std::function<void(const Vector &)>;
    using Boundary = std::function<void(size_t, size_t)>;
    using Speak = std::function<void(const DuplexReply &, const Stop &, const Audio &, const Boundary &)>;
    using Query = std::function<DuplexJson(const DuplexJson &, const Stop &)>;
    struct Event { std::string name; DuplexJson data; };
    struct Limits {
        size_t input_bytes = 8 * 1024 * 1024;
        size_t fragments = 64;
        size_t history_bytes = 32 * 1024 * 1024;
        size_t events = 64;
        size_t event_bytes = 4 * 1024 * 1024;
        std::chrono::milliseconds partial_interval{100};
        uint64_t playback_window = 0; // samples; zero preserves legacy transports
        bool committed_history = false;
    };
    DuplexSession(std::string id, DuplexJson history, Think think, Speak speak, Limits limits, Query query = {});
    ~DuplexSession();
    void start();
    DuplexJson append(uint64_t sequence, const DuplexJson & content, bool final,
                      bool replace = false, const DuplexJson & visual = DuplexJson::object(), const std::string & history_text = "", int max_tokens = -1,
                      const std::string & context = "", bool record_input = true, const std::string & spoken_text = "");
    DuplexJson query(const DuplexJson & request, const Stop & disconnected);
    DuplexJson context(const DuplexJson & request);
    DuplexJson interrupt_input(int64_t sequence, const DuplexJson & content, bool record_input = true);
    DuplexJson cancel(const std::string & generation);
    DuplexJson acknowledge(const std::string & generation, uint64_t played_samples, bool final = true);
    bool next(Event & event, const Stop & disconnected);
    void close();
    bool is_closed() const { return closed.load(); }
    const std::string id;
private:
    struct Input {
        DuplexJson content = DuplexJson::array();
        uint64_t version = 0, sequence = 0;
        size_t bytes = 0;
        bool final = false;
        bool record_input = true;
        DuplexJson visual = DuplexJson::object();
        std::string history_text;
        std::string context;
        std::string spoken_text;
        int max_tokens = -1;
    };
    struct QueryJob {
        DuplexJson request, result;
        Stop disconnected;
        std::exception_ptr error;
        bool done = false;
    };
    struct Job {
        DuplexReply reply;
        std::string generation;
        uint64_t revision = 0, version = 0, sequence = 0;
        std::atomic<bool> cancelled{false};
        bool completed = false;
        bool synthesized = false;
        bool playback_final = false;
        uint64_t samples = 0, played_samples = 0;
        size_t assistant_index = 0;
        size_t committed_text = 0;
        std::vector<std::pair<size_t, size_t>> boundaries; // text bytes, sample offset
    };
    Think think;
    Speak speak;
    Query query_fn;
    std::shared_ptr<QueryJob> pending_query;
    Limits limits;
    DuplexJson history;
    std::unique_ptr<Input> input;
    std::shared_ptr<Job> ready, active;
    uint64_t expected_sequence = 0, revision = 0, generation = 0;
    uint64_t processed_version = 0, processed_revision = 0;
    std::atomic<uint64_t> epoch{0};
    std::atomic<bool> closed{false};
    std::mutex mutex, event_mutex, join_mutex;
    std::condition_variable changed, event_changed;
    std::deque<Event> events;
    size_t queued_bytes = 0;
    std::thread thinker, talker;
    bool emit(std::string name, DuplexJson data, const Stop & stop = {});
    void think_loop();
    void speech_loop();
    void fail(const std::string & error);
    void commit_played(const std::shared_ptr<Job> & job); // mutex held
};
}
