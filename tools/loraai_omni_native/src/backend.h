#pragma once
#include "omni.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <map>
#include <functional>

namespace lora_omni {
bool use_device_backend();
struct DeviceModel {
    ggml_backend_t backend = nullptr;
    ggml_context * context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_gallocr_t allocator = nullptr;
    std::map<std::string, ggml_tensor *> tensors;
    bool paged = false;
    bool float_weights = false;
    std::string path;
    struct Source { size_t offset, bytes; ggml_type type; };
    std::map<std::string, Source> sources;
    mutable std::vector<ggml_backend_buffer_t> pages;
    DeviceModel(const std::string & path, bool float_weights);
    ~DeviceModel();
    ggml_tensor * weight(const std::string & name) const;
    ggml_tensor * experts(ggml_context * ctx, const std::string & name, const std::vector<int32_t> & ids);
    void release_pages();
};
ggml_backend_dev_t select_device();
size_t device_available_bytes();
void ensure_device_memory(size_t bytes);
size_t device_reserve_bytes();
std::shared_ptr<DeviceModel> device_model(const std::string & path, bool float_weights = false);
void release_device_models();
void release_unused_device_workspaces();
struct DeviceCacheScope { ~DeviceCacheScope() { release_device_models(); } };
struct DeviceCancellationScope {
    std::function<bool()> previous;
    explicit DeviceCancellationScope(std::function<bool()> stop);
    ~DeviceCancellationScope();
};
struct DeviceMemoryPressureScope {
    std::function<void()> previous;
    explicit DeviceMemoryPressureScope(std::function<void()> reclaim);
    ~DeviceMemoryPressureScope();
};
struct DeviceGraph {
    DeviceModel & model;
    ggml_context * ctx;
    ggml_cgraph * graph;
    std::vector<std::pair<ggml_tensor *, std::vector<char>>> inputs;
    explicit DeviceGraph(DeviceModel & model);
    ~DeviceGraph();
    ggml_tensor * input(const Vector & values, int width);
    ggml_tensor * integers(const std::vector<int32_t> & values);
    ggml_tensor * weight(const std::string & name) const;
    ggml_tensor * linear(const std::string & name, ggml_tensor * x);
    ggml_tensor * norm(const std::string & name, ggml_tensor * x);
    Vector run(ggml_tensor * output);
};
class DeviceDecoder {
    friend struct DeviceDecoderTest;
public:
    DeviceDecoder(const std::string & path, int capacity, bool predictor);
    ~DeviceDecoder();
    Vector step(const Vector & input);
    Vector logits(const Vector & input, int group = -1);
    Vector embedding(int id, int group = -1);
    void reset();
    void rewind(int prefix_length);
private:
    std::shared_ptr<DeviceModel> model;
    ggml_context * cache_context = nullptr;
    ggml_backend_buffer_t cache_buffer = nullptr;
    std::vector<ggml_tensor *> keys, values;
    int capacity = 0, max_capacity, position = 0, layers, kv_heads;
    void grow(int required);
    bool predictor;
    std::string prefix;
};
Vector device_project(const std::string & model, const Vector & input, const std::string & mode);
Vector device_codec_embedding(const std::string & model, const std::vector<Codes> & codes);
Vector device_codec_transformer(const std::string & model, const Vector & input);
Vector device_codec_upsample(const std::string & model, const Vector & input);
Vector device_codec_dac(const std::string & model, const Vector & input);
}
