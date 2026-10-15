#include "omni.h"
#include "backend.h"
#include "projection.inc"

namespace lora_omni {
static Vector project(const std::string & model, const Vector & input, int threads, const std::string & mode) {
    if (input.empty() || input.size() % INPUT) throw std::runtime_error("invalid Thinker embeddings");
    if (use_device_backend()) return device_project(model, input, mode);
    omni::Reader reader(model);
    const std::string stem = "talker." + mode + "_projection.";
    const auto w1 = read_bf16(reader, stem + "linear_fc1.weight", {INPUT, INTERMEDIATE});
    const auto b1 = read_bf16(reader, stem + "linear_fc1.bias", {INTERMEDIATE});
    const auto w2 = read_bf16(reader, stem + "linear_fc2.weight", {INTERMEDIATE, OUTPUT});
    const auto b2 = read_bf16(reader, stem + "linear_fc2.bias", {OUTPUT});
    return compute(input, w1, b1, w2, b2, INPUT, INTERMEDIATE, OUTPUT, int(input.size() / INPUT), threads).out;
}
Vector project_text(const std::string & model, const Vector & input, int threads) { return project(model, input, threads, "text"); }
Vector project_hidden(const std::string & model, const Vector & input, int threads) { return project(model, input, threads, "hidden"); }
}
