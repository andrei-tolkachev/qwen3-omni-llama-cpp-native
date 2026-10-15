#pragma once
#include "omni.h"

namespace lora_omni {
// Sliding attention state and causal convolution overlap, independent of reply length.
class CodecStream {
public:
    explicit CodecStream(std::string model): model(std::move(model)) {}
    Vector push(const std::vector<Codes> & codes);
private:
    std::string model;
    std::array<Vector, 8> keys, values;
    Vector up_history, dac_history;
    int position = 0;
    size_t emitted = 0;
    Vector transform(const Vector & input);
};
}
