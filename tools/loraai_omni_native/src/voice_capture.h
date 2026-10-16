#pragma once
#include "voice.h"
#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <utility>

namespace lora_omni {
struct VoiceCapture {
    std::atomic<bool> active{false};
    std::vector<CapturedRow> rows;
    std::string error;
    bool role_pending = false, in_user = false, overflow = false;
    size_t capacity = 0;

    void reset(size_t thinker_context) {
        if (!thinker_context) throw std::invalid_argument("invalid Thinker context capacity");
        capacity = std::min(thinker_context, speech_context_capacity);
        rows.clear(); error.clear(); role_pending = in_user = overflow = false;
    }
    void push(CapturedRow row) {
        if (row.token == 151644) {
            role_pending = true;
            in_user = false;
            return;
        }
        if (role_pending) {
            role_pending = false;
            in_user = row.token == 872;
            if (in_user) {
                // Latest user states attend to the full Thinker dialogue.
                rows.clear(); overflow = false;
                rows.push_back({151644, {}});
            }
        }
        if (!in_user) return;
        if (rows.size() >= capacity) { overflow = true; return; }
        rows.push_back(std::move(row));
    }
};
}
