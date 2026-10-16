#include "backend.h"
#include "reader.inc"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>

namespace lora_omni {
namespace {
size_t resident_bytes = 0;
void initialize_backends() {
    // Registry mutation belongs to initialization, never a speech graph running
    // concurrently with llama's Thinker/backend enumeration.
    static std::once_flag initialized;
    std::call_once(initialized, [] { ggml_backend_load_all(); });
}
thread_local std::function<bool()> cancellation;
thread_local std::function<void()> memory_pressure;
void check_cancelled() {
    if (cancellation && cancellation()) throw std::runtime_error("voice_cancelled");
}
size_t setting_bytes(const char * name, size_t fallback) {
    const char * value = std::getenv(name);
    if (!value) return fallback;
    size_t length = 0;
    const auto mib = std::stoull(value, &length);
    if (length != std::strlen(value) || mib > 1048576) throw std::runtime_error(std::string("invalid ") + name);
    return size_t(mib) * 1024 * 1024;
}
}
DeviceCancellationScope::DeviceCancellationScope(std::function<bool()> stop): previous(std::move(cancellation)) { cancellation = std::move(stop); }
DeviceCancellationScope::~DeviceCancellationScope() { cancellation = std::move(previous); }
DeviceMemoryPressureScope::DeviceMemoryPressureScope(std::function<void()> reclaim): previous(std::move(memory_pressure)) { memory_pressure = std::move(reclaim); }
DeviceMemoryPressureScope::~DeviceMemoryPressureScope() { memory_pressure = std::move(previous); }
ggml_backend_dev_t select_device() {
    initialize_backends();
    if (const char * name = std::getenv("OMNI_DEVICE")) {
        auto * device = ggml_backend_dev_by_name(name);
        if (!device || (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_IGPU))
            throw std::runtime_error("OMNI_DEVICE must name an available GPU device");
        return device;
    }
    // Keep the selected device stable while component weights are being loaded.
    static ggml_backend_dev_t chosen = [] {
        ggml_backend_dev_t best = nullptr;
        size_t greatest = 0;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto * device = ggml_backend_dev_get(i);
            const auto type = ggml_backend_dev_type(device);
            if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
            size_t available, total;
            ggml_backend_dev_memory(device, &available, &total);
            if (!best || available > greatest) { best = device; greatest = available; }
        }
        return best;
    }();
    if (!chosen) throw std::runtime_error("no GPU backend is available for Omni speech");
    return chosen;
}
size_t device_reserve_bytes() { return setting_bytes("OMNI_GPU_RESERVE_MB", 512ULL * 1024 * 1024); }
size_t device_available_bytes() {
    size_t available, total;
    ggml_backend_dev_memory(select_device(), &available, &total);
    const size_t budget = setting_bytes("OMNI_GPU_BUDGET_MB", total);
    available = std::min(available, budget > resident_bytes ? budget - resident_bytes : size_t(0));
    return available;
}
static void release_unused_device_models();
void ensure_device_memory(size_t bytes) {
    auto fits = [&] { return bytes + device_reserve_bytes() <= device_available_bytes(); };
    if (!fits()) release_unused_device_workspaces();
    if (!fits()) release_unused_device_models();
    if (!fits() && memory_pressure) memory_pressure();
    check_cancelled();
    if (!fits()) throw std::runtime_error("Omni GPU working-set memory budget exceeded");
}
bool use_device_backend() {
    const char * setting = std::getenv("OMNI_BACKEND");
    if (!setting || std::string(setting) == "gpu") return true;
    if (std::string(setting) == "cpu") return false;
    throw std::runtime_error("OMNI_BACKEND must be gpu or cpu");
}
DeviceModel::DeviceModel(const std::string & path, bool float_weights): float_weights(float_weights), path(path) {
    bool accounted = false;
    try {
        initialize_backends();
        backend = ggml_backend_dev_init(select_device(), nullptr);
        if (!backend) throw std::runtime_error("Omni requires a GPU backend; set OMNI_BACKEND=cpu for the reference path");
        omni::Reader reader(path);
        const int64_t count = reader.tensors().size();
        context = ggml_init({size_t(count + 1) * ggml_tensor_overhead(), nullptr, true});
        if (!context) throw std::runtime_error("cannot create Omni weight context");
        for (int64_t i = 0; i < count; ++i) {
            check_cancelled();
            const auto & info = reader.tensors()[i];
            int64_t ne[4] = {1, 1, 1, 1};
            for (size_t j = 0; j < info.dims.size(); ++j) ne[j] = info.dims[j];
            const auto type = (float_weights || (ne[1] == 1 && ne[2] == 1)) ? GGML_TYPE_F32 : ggml_type(info.type);
            auto * tensor = ggml_new_tensor(context, type, 4, ne);
            const std::string name = info.name;
            ggml_set_name(tensor, ("omni_weight_" + std::to_string(i)).c_str());
            tensors.emplace(name, tensor);
            sources.emplace(name, Source{size_t(reader.data_start() + info.offset), size_t(info.bytes), ggml_type(info.type)});
        }
        const size_t required = ggml_backend_alloc_ctx_tensors_from_buft_size(context, ggml_backend_get_default_buffer_type(backend));
        const size_t available = device_available_bytes(), reserve = device_reserve_bytes();
        paged = available <= reserve || required > available - reserve;
        if (!paged) {
        buffer = ggml_backend_alloc_ctx_tensors(context, backend);
        if (!buffer) throw std::runtime_error("GPU memory exhausted loading Omni weights");
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (int64_t i = 0; i < count; ++i) {
            check_cancelled();
            const auto & info = reader.tensors()[i];
            auto * target = weight(info.name);
            const auto source_type = ggml_type(info.type);
            auto raw = reader.read_region(info.offset, info.bytes);
            if (source_type == target->type) ggml_backend_tensor_set(target, raw.data(), 0, raw.size());
            else {
                const auto * traits = ggml_get_type_traits(source_type);
                if (!traits->to_float) throw std::runtime_error("unsupported Omni weight type");
                Vector data(ggml_nelements(target));
                traits->to_float(raw.data(), data.data(), data.size());
                ggml_backend_tensor_set(target, data.data(), 0, data.size() * sizeof(float));
            }
        }
        resident_bytes += ggml_backend_buffer_get_size(buffer);
        accounted = true;
        }
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!allocator) throw std::runtime_error("cannot create Omni graph allocator");
        std::cerr << "Omni weights: " << ggml_backend_name(backend) << ", "
                  << required / (1024. * 1024.) << " MiB, " << (paged ? "layer paging, " : "resident, ") << path << '\n';
    } catch (...) {
        if (accounted) resident_bytes -= ggml_backend_buffer_get_size(buffer);
        if (allocator) ggml_gallocr_free(allocator);
        if (buffer) ggml_backend_buffer_free(buffer);
        if (context) ggml_free(context);
        if (backend) ggml_backend_free(backend);
        throw;
    }
}
DeviceModel::~DeviceModel() {
    release_pages();
    ggml_gallocr_free(allocator);
    if (buffer) resident_bytes -= ggml_backend_buffer_get_size(buffer);
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    ggml_backend_free(backend);
}
ggml_tensor * DeviceModel::weight(const std::string & name) const {
    const auto found = tensors.find(name);
    if (found == tensors.end()) throw std::runtime_error("missing Omni tensor: " + name);
    auto * tensor = found->second;
    if (paged && !tensor->buffer) {
        auto * buft = ggml_backend_get_default_buffer_type(backend);
        const size_t size = ggml_backend_buft_get_alloc_size(buft, tensor);
        ensure_device_memory(size);
        auto * page = ggml_backend_buft_alloc_buffer(buft, size);
        if (!page) throw std::runtime_error("cannot allocate Omni GPU weight page");
        if (ggml_backend_tensor_alloc(page, tensor, ggml_backend_buffer_get_base(page)) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(page); throw std::runtime_error("cannot bind Omni weight page");
        }
        pages.push_back(page);
        resident_bytes += ggml_backend_buffer_get_size(page);
        ggml_backend_buffer_set_usage(page, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        const auto & source = sources.at(name);
        std::ifstream file(path, std::ios::binary);
        std::vector<char> raw(source.bytes);
        file.seekg(source.offset); file.read(raw.data(), raw.size());
        if (!file) throw std::runtime_error("cannot read Omni GPU weight page");
        if (source.type == tensor->type) ggml_backend_tensor_set(tensor, raw.data(), 0, raw.size());
        else {
            Vector converted(ggml_nelements(tensor));
            const auto * traits = ggml_get_type_traits(source.type);
            if (!traits->to_float) throw std::runtime_error("unsupported Omni page type");
            traits->to_float(raw.data(), converted.data(), converted.size());
            ggml_backend_tensor_set(tensor, converted.data(), 0, converted.size() * sizeof(float));
        }
    }
    return tensor;
}
void DeviceModel::release_pages() {
    if (!paged) return;
    for (auto * page : pages) {
        resident_bytes -= ggml_backend_buffer_get_size(page);
        ggml_backend_buffer_free(page);
    }
    pages.clear();
    for (auto & entry : tensors) { entry.second->buffer = nullptr; entry.second->data = nullptr; }
}
ggml_tensor * DeviceModel::experts(ggml_context * ctx, const std::string & name, const std::vector<int32_t> & ids) {
    const auto found = tensors.find(name);
    if (found == tensors.end() || ids.empty()) throw std::runtime_error("invalid Omni expert group");
    auto * full = found->second;
    const auto & source = sources.at(name);
    const size_t slice = source.bytes / size_t(full->ne[2]);
    auto * tensor = ggml_new_tensor_3d(ctx, full->type, full->ne[0], full->ne[1], ids.size());
    const size_t required = ggml_backend_buft_get_alloc_size(ggml_backend_get_default_buffer_type(backend), tensor);
    ensure_device_memory(required);
    auto * page = ggml_backend_alloc_buffer(backend, required);
    if (!page) throw std::runtime_error("cannot allocate selected Omni experts");
    if (ggml_backend_tensor_alloc(page, tensor, ggml_backend_buffer_get_base(page)) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(page); throw std::runtime_error("cannot bind selected Omni experts");
    }
    ggml_backend_buffer_set_usage(page, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    pages.push_back(page); resident_bytes += ggml_backend_buffer_get_size(page);
    std::ifstream file(path, std::ios::binary);
    std::vector<char> raw(slice);
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] < 0 || ids[i] >= full->ne[2]) throw std::runtime_error("invalid selected Omni expert");
        file.seekg(source.offset + size_t(ids[i]) * slice); file.read(raw.data(), raw.size());
        if (!file) throw std::runtime_error("cannot read selected Omni expert");
        ggml_backend_tensor_set(tensor, raw.data(), i * slice, slice);
    }
    return tensor;
}
namespace {
auto & model_cache() {
    static std::map<std::pair<std::string, bool>, std::shared_ptr<DeviceModel>> loaded;
    return loaded;
}
}
void release_device_models() { model_cache().clear(); }
static void release_unused_device_models() {
    auto & loaded = model_cache();
    for (auto it = loaded.begin(); it != loaded.end();) {
        if (it->second.use_count() == 1) it = loaded.erase(it);
        else ++it;
    }
}
void release_unused_device_workspaces() {
    for (auto & item : model_cache()) {
        if (item.second.use_count() != 1) continue;
        auto & model = *item.second;
        auto replacement = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!replacement) throw std::runtime_error("cannot reset Omni graph allocator");
        ggml_gallocr_free(model.allocator);
        model.allocator = replacement;
    }
}
std::shared_ptr<DeviceModel> device_model(const std::string & path, bool float_weights) {
    initialize_backends();
    // Only the speech lane owns this cache. Thinker has independent backends.
    auto & loaded = model_cache();
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    const auto key = std::make_pair(path, float_weights);
    auto found = loaded.find(key);
    if (found != loaded.end()) return found->second;
    auto model = std::make_shared<DeviceModel>(path, float_weights);
    loaded.emplace(key, model);
    return model;
}
DeviceGraph::DeviceGraph(DeviceModel & model): model(model) {
    ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
    if (!ctx) throw std::runtime_error("cannot create Omni graph");
    graph = ggml_new_graph_custom(ctx, 16384, false);
}
DeviceGraph::~DeviceGraph() { model.release_pages(); ggml_free(ctx); }
ggml_tensor * DeviceGraph::input(const Vector & data, int width) {
    if (data.empty() || width < 1 || data.size() % width) throw std::runtime_error("invalid Omni input shape");
    auto * tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, data.size() / width);
    ggml_set_input(tensor);
    std::vector<char> raw(data.size() * sizeof(float));
    std::memcpy(raw.data(), data.data(), raw.size());
    inputs.emplace_back(tensor, std::move(raw));
    return tensor;
}
ggml_tensor * DeviceGraph::integers(const std::vector<int32_t> & data) {
    auto * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, data.size());
    ggml_set_input(tensor);
    std::vector<char> raw(data.size() * sizeof(int32_t));
    std::memcpy(raw.data(), data.data(), raw.size());
    inputs.emplace_back(tensor, std::move(raw));
    return tensor;
}
ggml_tensor * DeviceGraph::weight(const std::string & name) const { return model.weight(name); }
ggml_tensor * DeviceGraph::linear(const std::string & name, ggml_tensor * x) {
    auto * result = ggml_mul_mat(ctx, weight(name), x);
    ggml_mul_mat_set_prec(result, GGML_PREC_F32);
    return result;
}
ggml_tensor * DeviceGraph::norm(const std::string & name, ggml_tensor * x) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, 1.e-6f), weight(name));
}
Vector DeviceGraph::run(ggml_tensor * output) {
    check_cancelled();
    output = ggml_cont(ctx, output);
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto * node = ggml_graph_node(graph, i);
        if (!ggml_backend_supports_op(model.backend, node))
            throw std::runtime_error(std::string("Omni GPU operation unsupported: ") + ggml_op_name(node->op));
    }
    if (model.float_weights) {
        // Code2Wav activation memory grows with the actual frame count.
        // Measure before allocating; keep the current decoder alive while
        // releasing cached Talker/Predictor weights and, if needed, Thinker.
        size_t required = 0;
        ggml_gallocr_reserve_n_size(model.allocator, graph, nullptr, nullptr, &required);
        const size_t allocated = ggml_gallocr_get_buffer_size(model.allocator, 0);
        const size_t additional = required > allocated ? required - allocated : 0;
        auto fits = [&] { return additional + device_reserve_bytes() <= device_available_bytes(); };
        if (!fits()) release_unused_device_models();
        if (!fits() && memory_pressure) memory_pressure();
        check_cancelled();
        if (!fits()) throw std::runtime_error("Omni GPU decode memory budget exceeded");
        // Measuring can release an undersized old buffer; reserve it again.
        if (!ggml_gallocr_reserve(model.allocator, graph)) throw std::runtime_error("Omni GPU graph allocation failed");
    }
    if (!ggml_gallocr_alloc_graph(model.allocator, graph)) throw std::runtime_error("Omni GPU graph allocation failed");
    for (auto & input : inputs) ggml_backend_tensor_set(input.first, input.second.data(), 0, input.second.size());
    if (ggml_backend_graph_compute(model.backend, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("Omni GPU graph execution failed");
    Vector result(ggml_nelements(output));
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    check_cancelled();
    for (float value : result) if (!std::isfinite(value)) throw std::runtime_error("nonfinite Omni GPU output");
    return result;
}
}
