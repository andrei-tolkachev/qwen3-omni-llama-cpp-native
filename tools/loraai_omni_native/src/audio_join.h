#pragma once
#include "omni.h"
#include <algorithm>
#include <functional>
namespace lora_omni {
// Keep only 10 ms of PCM. Overlap complete phrase boundaries, never add silence.
class AudioJoin {
    Vector tail;
    bool boundary = false;
    std::function<void(const Vector &)> emit;
public:
    explicit AudioJoin(std::function<void(const Vector &)> sink): emit(std::move(sink)) {}
    void next_phrase() { boundary = true; }
    void push(const Vector & pcm) {
        Vector joined = tail;
        size_t overlap = boundary ? std::min({size_t(240), joined.size(), pcm.size()}) : 0;
        for (size_t i = 0; i < overlap; ++i) {
            const float weight = float(i + 1) / float(overlap + 1);
            joined[joined.size() - overlap + i] = joined[joined.size() - overlap + i] * (1.f - weight) + pcm[i] * weight;
        }
        joined.insert(joined.end(), pcm.begin() + overlap, pcm.end());
        boundary = false;
        const size_t retain = std::min(size_t(240), joined.size());
        const size_t count = joined.size() - retain;
        if (count) emit(Vector(joined.begin(), joined.begin() + count));
        tail.assign(joined.begin() + count, joined.end());
    }
    void finish() { if (!tail.empty()) emit(tail); tail.clear(); }
};
}
