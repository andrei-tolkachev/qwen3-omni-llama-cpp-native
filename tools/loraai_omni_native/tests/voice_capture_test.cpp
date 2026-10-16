#include "voice_capture.h"
#include <cassert>
#include <iostream>

int main() {
    lora_omni::VoiceCapture capture;
    auto role = [&](int token) { capture.push({151644, {}}); capture.push({token, {}}); };
    capture.reset(65536);
    role(8948);
    for (int i = 0; i < 13000; ++i) capture.push({42, {}});
    assert(capture.rows.empty());
    role(872);
    for (int i = 0; i < 12000; ++i) capture.push({100 + i, {}});
    lora_omni::Vector hidden(2048, .25f);
    capture.push({-1, hidden});
    capture.push({151645, {}});
    role(77091);
    capture.push({71, {}});
    assert(!capture.overflow && capture.rows.size() == 12004);
    assert(capture.rows.front().token == 151644 && capture.rows[1].token == 872);
    for (int i = 0; i < 12000; ++i) assert(capture.rows[2 + i].token == 100 + i);
    assert(capture.rows[12002].hidden == hidden && capture.rows.back().token == 151645);
    role(872);
    capture.push({19, {}});
    assert(capture.rows.size() == 3 && capture.rows.back().token == 19);

    capture.reset(4096);
    role(872);
    for (int i = 0; i < 4094; ++i) capture.push({42, {}});
    assert(capture.rows.size() == 4096 && !capture.overflow);
    capture.push({43, {}});
    assert(capture.overflow && capture.rows.size() == 4096);
    role(872);
    assert(!capture.overflow && capture.rows.size() == 2);
    capture.reset(131072);
    assert(capture.capacity == lora_omni::speech_context_capacity);
    role(872);
    for (size_t i = 2; i < capture.capacity; ++i) capture.push({42, {}});
    assert(!capture.overflow);
    capture.push({43, {}});
    assert(capture.overflow && capture.rows.size() == 65536);
    std::cout << "PASS: long conditioning, lossless multimodal rows, latest user and actual context bounds\n";
}
