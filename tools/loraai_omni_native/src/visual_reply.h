#pragma once
#include <functional>
#include <string>
#include "json.h"

namespace lora_omni {
// Only this explicit protocol is removed from speech. A partial opening tag is
// buffered across token boundaries; malformed/truncated commands never become
// pointer events or spoken technical markup.
class VisualReply {
    using json = common_json;
    std::string pending_, frame_id_;
    bool command_ = false;
    bool allow_screen_request_ = false;
    std::function<void(const std::string &, const json &)> emit_;
    static constexpr const char * open_ = "<lora_pointer>";
    static constexpr const char * close_ = "</lora_pointer>";
    void text(const std::string & value) {
        if (value.empty()) return;
        reply += value;
        if (emit_) emit_("delta", {{"text", value}});
    }
    void pointer(const std::string & raw) {
        try {
            auto p = json::parse(raw);
            if (allow_screen_request_ && frame_id_.empty() && actions.empty() && p.is_object()
                && p.size() == 1 && p.value("request_screen", false)) {
                json action = {{"type", "request_screen"}};
                actions.push_back(action);
                if (emit_) emit_("live_action", action);
                return;
            }
            if (frame_id_.empty() || pointers.size() >= 3) return;
            if (!p.is_object() || !p.contains("x") || !p.contains("y")) return;
            for (const auto & key : {"x", "y", "width", "height"}) {
                if (!p.contains(key)) continue;
                if (!p[key].is_number() || !(p[key].get<double>() >= 0 && p[key].get<double>() <= 1)) return;
            }
            if (p.contains("width") != p.contains("height")) return;
            if (p.contains("width") && (p["x"].get<double>() + p["width"].get<double>() > 1 || p["y"].get<double>() + p["height"].get<double>() > 1)) return;
            auto label = p.value("label", std::string{});
            if (label.size() > 240) return;
            json safe = {{"frame_id", frame_id_}, {"x", p["x"]}, {"y", p["y"]}, {"label", label}};
            if (p.contains("width")) { safe["width"] = p["width"]; safe["height"] = p["height"]; }
            pointers.push_back(safe);
            if (emit_) emit_("pointer", safe);
        } catch (...) { /* Invalid model output is not an executable command. */ }
    }
public:
    std::string reply;
    json pointers = json::array();
    json actions = json::array();
    VisualReply(std::string frame_id, decltype(emit_) emit, bool allow_screen_request = false)
        : frame_id_(std::move(frame_id)), allow_screen_request_(allow_screen_request), emit_(std::move(emit)) {}
    void push(const std::string & part) {
        pending_ += part;
        for (;;) {
            if (command_) {
                auto end = pending_.find(close_);
                if (end == std::string::npos) {
                    if (pending_.size() > 4096) pending_.erase(0, pending_.size() - 16);
                    return;
                }
                pointer(pending_.substr(0, end));
                pending_.erase(0, end + std::string(close_).size()); command_ = false;
            } else {
                auto begin = pending_.find(open_);
                if (begin != std::string::npos) {
                    text(pending_.substr(0, begin));
                    pending_.erase(0, begin + std::string(open_).size()); command_ = true;
                } else {
                    size_t keep = 0;
                    const std::string marker(open_);
                    for (size_t n = 1; n < marker.size() && n <= pending_.size(); ++n)
                        if (pending_.compare(pending_.size() - n, n, marker, 0, n) == 0) keep = n;
                    text(pending_.substr(0, pending_.size() - keep));
                    pending_.erase(0, pending_.size() - keep); return;
                }
            }
        }
    }
    void finish() { if (!command_ && pending_.find('<') == std::string::npos) text(pending_); pending_.clear(); }
};
}
