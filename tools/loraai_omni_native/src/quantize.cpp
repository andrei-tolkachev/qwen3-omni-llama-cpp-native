#include "ggml.h"
#include "../vendor/reader.inc"
#include <cstring>

// GGUF storage supports full HF tensor names. The generic GGML GGUF loader
// limits names to GGML_MAX_NAME, so use the speech reader/writer for the header
// and the public GGML API for numerical quantization. No llama core patches.
namespace {
void number(std::ostream & out, uint64_t value, unsigned width) {
    for (unsigned i = 0; i < width; ++i) out.put(char(value >> (8 * i)));
}
void string(std::ostream & out, const std::string & value) {
    number(out, value.size(), 8); out.write(value.data(), value.size());
}
void header(std::ostream & out, const std::map<std::string, omni::Meta> & meta,
            const std::vector<omni::Tensor> & tensors) {
    out.write("GGUF", 4); number(out, 3, 4); number(out, tensors.size(), 8); number(out, meta.size(), 8);
    for (const auto & entry : meta) {
        string(out, entry.first); const auto & v = entry.second; number(out, v.type, 4);
        if (v.type == 8) string(out, v.text);
        else if (v.type == 7) number(out, v.flag, 1);
        else number(out, v.number, v.type == 4 ? 4 : 8);
    }
    for (const auto & t : tensors) {
        string(out, t.name); number(out, t.dims.size(), 4);
        for (auto dim : t.dims) number(out, dim, 8);
        number(out, t.type, 4); number(out, t.offset, 8);
    }
}
}
int main(int argc, char ** argv) {
    fs::path partial; bool created = false;
    try {
        if (argc != 3) throw std::runtime_error("usage: omni-quantize INPUT_BF16.gguf OUTPUT_Q4.gguf");
        const fs::path input(argv[1]), output(argv[2]);
        partial = output.string() + ".partial";
        if (fs::exists(output) || fs::exists(partial)) throw std::runtime_error("output already exists");
        omni::Reader reader(input);
        if (omni::require_string(reader, "loraai.omni.component") != "talker" ||
            omni::require_string(reader, "general.architecture") != omni::EXPECTED_ARCH)
            throw std::runtime_error("expected packed Talker BF16 model");
        auto tensors = reader.tensors(); auto meta = reader.metadata();
        size_t changed = 0; uint64_t offset = 0;
        const size_t alignment = meta.count("general.alignment") ? meta.at("general.alignment").number : 32;
        for (auto & tensor : tensors) {
            if (tensor.type != GGML_TYPE_BF16) throw std::runtime_error("input must be entirely BF16; requantization is refused");
            if (tensor.name.find(".mlp.experts_packed.") != std::string::npos) {
                if (tensor.dims.size() != 3 || tensor.dims[2] != 128 || tensor.dims[0] % ggml_blck_size(GGML_TYPE_Q4_0))
                    throw std::runtime_error("unsupported packed expert geometry");
                tensor.type = tensor.dims[0] % ggml_blck_size(GGML_TYPE_Q4_K) == 0 ? GGML_TYPE_Q4_K : GGML_TYPE_Q4_0;
                tensor.bytes = ggml_row_size(static_cast<ggml_type>(tensor.type), tensor.dims[0]) * tensor.dims[1] * tensor.dims[2];
                ++changed;
            }
            tensor.offset = offset; offset = omni::round_up(offset + tensor.bytes, alignment);
        }
        if (changed != 60) throw std::runtime_error("expected 60 packed expert tensors");
        // There is no standard GGUF file-type enum for this custom mixed
        // profile. Tensor types and the explicit profile are authoritative.
        meta.erase("general.file_type");
        meta["general.name"] = {8, "Qwen3-Omni Talker: mixed Q4 experts", 0, false};
        meta["general.description"] = {8, "Q4_K gate/up, Q4_0 down, BF16 sensitive tensors; requires native Omni extension", 0, false};
        meta["loraai.omni.quantization"] = {8, "Q4_K-gate-up-Q4_0-down-BF16-sensitive-v1", 0, false};
        meta["general.quantization_version"] = {4, "", GGML_QNT_VERSION, false};
        std::ostringstream encoded(std::ios::binary); header(encoded, meta, tensors);
        const auto bytes_header = encoded.str();
        std::ofstream file(partial, std::ios::binary); created = bool(file);
        const std::vector<char> zeros(alignment, 0);
        file.write(bytes_header.data(), bytes_header.size());
        file.write(zeros.data(), omni::round_up(bytes_header.size(), alignment) - bytes_header.size());
        uint64_t bytes_out = 0;
        for (size_t i = 0; i < tensors.size(); ++i) {
            const auto & before = reader.tensors()[i]; const auto & after = tensors[i];
            auto bytes = reader.read_region(before.offset, before.bytes);
            if (after.type != GGML_TYPE_BF16) {
                const int64_t width = after.dims[0], rows = after.dims[1] * after.dims[2];
                // At most one packed tensor is expanded; inference keeps Q4 on GPU.
                std::vector<float> values(size_t(width * rows));
                ggml_get_type_traits(GGML_TYPE_BF16)->to_float(bytes.data(), values.data(), width * rows);
                bytes.resize(after.bytes);
                const auto written = ggml_quantize_chunk(static_cast<ggml_type>(after.type), values.data(), bytes.data(), 0, rows, width, nullptr);
                if (written != bytes.size()) throw std::runtime_error("quantized size mismatch");
            }
            file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
            file.write(zeros.data(), omni::round_up(bytes.size(), alignment) - bytes.size());
            if (!file) throw std::runtime_error("output write failed");
            bytes_out += bytes.size();
        }
        file.close(); if (!file) throw std::runtime_error("output flush failed");
        omni::Reader verified(partial);
        if (verified.tensors().size() != tensors.size() || verified.payload_bytes() != bytes_out)
            throw std::runtime_error("output validation failed");
        fs::rename(partial, output); created = false;
        ggml_quantize_free();
        std::cout << "{\"quantized_tensors\":" << changed << ",\"source_payload_bytes\":" << reader.payload_bytes()
                  << ",\"output_payload_bytes\":" << bytes_out << "}\n";
        return 0;
    } catch (const std::exception & error) {
        if (created) { std::error_code ignored; fs::remove(partial, ignored); }
        std::cerr << error.what() << '\n'; return 1;
    }
}
