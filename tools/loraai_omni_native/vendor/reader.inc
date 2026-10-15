// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andrei Tolkachev
// Derived from the user-provided, BF16-verified LoraAI kernels. See provenance.json.
// Modified: block-aware Q4_K/Q4_0 tensor sizing for mixed-precision speech GGUF.
#pragma once
#include "ggml.h"
// LoraAI Qwen3 Omni S1-B2: standalone read-only staging GGUF tensor indexer.
// Does NOT execute Talker or load tensors into GPU/CPU memory.
// See docs/CONTRACT.md before integrating with llama.cpp GGML graph code.
#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace { namespace omni {
constexpr std::uint32_t GGUF_BF16 = 30;
constexpr std::uint64_t EXPECTED_PAYLOAD = 6649192960ULL;
constexpr std::size_t EXPECTED_COUNT = 417;
constexpr std::size_t EXPECTED_GROUPS = 60;
constexpr const char * EXPECTED_ARCH = "qwen3_omni_talker_experts_packed_staging";

struct Failure : std::runtime_error { using runtime_error::runtime_error; };

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) throw Failure("uint64 offset overflow");
    return a + b;
}
std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b) {
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a) throw Failure("uint64 tensor size overflow");
    return a * b;
}
std::uint64_t round_up(std::uint64_t a, std::uint64_t n) {
    if (!n || (n & (n - 1)) || n > 4096) throw Failure("invalid GGUF alignment");
    return checked_add(a, n - 1) & ~(n - 1);
}

struct Meta {
    std::uint32_t type = 0;
    std::string text;
    std::uint64_t number = 0;
    bool flag = false;
};
struct Tensor {
    std::string name;
    std::vector<std::uint64_t> dims; // GGML order: [in, out, experts]
    std::uint32_t type = 0;
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
};
struct Layout {
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
    std::vector<std::uint64_t> dims;
};
struct Baseline {
    std::string source_index_sha;
    std::string config_sha;
    std::string llama_commit;
    std::map<std::string, Layout> expected;
};

class Reader {
public:
    explicit Reader(const fs::path & path) : path_(path), f_(path, std::ios::binary) {
        if (!f_) throw Failure("cannot open model: " + path.string());
        file_size_ = fs::file_size(path);
        if (file_size_ < 24) throw Failure("GGUF file too small");
        read_header();
    }
    const std::vector<Tensor> & tensors() const { return tensors_; }
    const std::map<std::string, Meta> & metadata() const { return metadata_; }
    std::uint64_t data_start() const { return data_start_; }
    std::uint64_t file_size() const { return file_size_; }
    std::uint64_t payload_bytes() const {
        std::uint64_t total = 0;
        for (const auto & t : tensors_) total = checked_add(total, t.bytes);
        return total;
    }
    const Tensor & get_tensor(const std::string & name) const {
        const auto it = names_.find(name);
        if (it == names_.end()) throw Failure("missing tensor: " + name);
        return tensors_[it->second];
    }
    std::vector<std::uint8_t> read_region(std::uint64_t rel_offset, std::size_t count) {
        const auto absolute = checked_add(data_start_, rel_offset);
        if (absolute > file_size_ || count > file_size_ - absolute) throw Failure("read region outside GGUF");
        f_.clear();
        f_.seekg(static_cast<std::streamoff>(absolute), std::ios::beg);
        if (!f_) throw Failure("cannot seek in GGUF");
        std::vector<std::uint8_t> result(count);
        f_.read(reinterpret_cast<char *>(result.data()), static_cast<std::streamsize>(count));
        if (static_cast<std::size_t>(f_.gcount()) != count) throw Failure("truncated GGUF tensor region");
        return result;
    }
private:
    fs::path path_;
    std::ifstream f_;
    std::uint64_t file_size_ = 0;
    std::uint64_t data_start_ = 0;
    std::uint64_t at_ = 0;
    std::map<std::string, Meta> metadata_;
    std::vector<Tensor> tensors_;
    std::map<std::string, std::size_t> names_;

    void fetch(void * p, std::size_t n) {
        if (at_ > file_size_ || n > file_size_ - at_) throw Failure("truncated GGUF header");
        f_.read(static_cast<char *>(p), static_cast<std::streamsize>(n));
        if (!f_) throw Failure("GGUF header read error");
        at_ += n;
    }
    std::uint64_t integer(unsigned bytes) {
        std::uint8_t raw[8]{};
        if (bytes > 8) throw Failure("unsupported integer width");
        fetch(raw, bytes);
        std::uint64_t value = 0;
        for (unsigned i = 0; i < bytes; ++i) value |= std::uint64_t(raw[i]) << (8 * i);
        return value;
    }
    std::string text() {
        const auto n = integer(8);
        if (n > (1u << 20)) throw Failure("GGUF string exceeds 1 MiB");
        std::string s(static_cast<std::size_t>(n), '\0');
        if (n) fetch(s.data(), static_cast<std::size_t>(n));
        return s;
    }
    void read_header() {
        char magic[4];
        fetch(magic, 4);
        if (std::string(magic, 4) != "GGUF") throw Failure("not a GGUF model");
        if (integer(4) != 3) throw Failure("GGUF version other than v3 unsupported");
        const auto tensor_count = integer(8), kv_count = integer(8);
        if (tensor_count > 200000 || kv_count > 4096) throw Failure("suspicious GGUF header count");
        for (std::uint64_t i = 0; i < kv_count; ++i) {
            const auto key = text();
            const auto type = integer(4);
            Meta value; value.type = static_cast<std::uint32_t>(type);
            // S1-A and S1-B1 staging deliberately use only these GGUF metadata types.
            if (type == 8) value.text = text();
            else if (type == 7) value.flag = integer(1) != 0;
            else if (type == 4) value.number = integer(4);
            else if (type == 10) value.number = integer(8);
            else throw Failure("unsupported GGUF metadata type for staging key " + key);
            if (!metadata_.emplace(key, std::move(value)).second) throw Failure("duplicate GGUF metadata key " + key);
        }
        for (std::uint64_t i = 0; i < tensor_count; ++i) {
            Tensor t;
            t.name = text();
            const auto rank = integer(4);
            if (rank < 1 || rank > 4) throw Failure("unexpected GGUF tensor rank: " + t.name);
            t.bytes = 1;
            for (std::uint64_t d = 0; d < rank; ++d) {
                const auto dim = integer(8);
                if (dim == 0) throw Failure("zero tensor dimension: " + t.name);
                t.dims.push_back(dim);
                t.bytes = checked_mul(t.bytes, dim);
            }
            t.type = static_cast<std::uint32_t>(integer(4));
            t.offset = integer(8);
            // Speech conversion keeps sensitive tensors in BF16; packed expert
            // matrices may use Q4_K. Compute storage in blocks, not elements.
            if (t.type != GGUF_BF16 && t.type != GGML_TYPE_Q4_K && t.type != GGML_TYPE_Q4_0)
                throw Failure("unsupported speech tensor type: " + t.name);
            const auto type = static_cast<ggml_type>(t.type);
            if (t.dims[0] % ggml_blck_size(type)) throw Failure("unaligned quantized row: " + t.name);
            t.bytes = checked_mul(t.bytes / t.dims[0], ggml_row_size(type, t.dims[0]));
            if (!names_.emplace(t.name, tensors_.size()).second) throw Failure("duplicate tensor name: " + t.name);
            tensors_.push_back(std::move(t));
        }
        std::uint64_t alignment = 32;
        const auto it = metadata_.find("general.alignment");
        if (it != metadata_.end()) {
            if (it->second.type != 4) throw Failure("alignment GGUF type must be uint32");
            alignment = it->second.number;
        }
        data_start_ = round_up(at_, alignment);
        if (data_start_ > file_size_) throw Failure("missing GGUF data region");
        std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
        for (const auto & t : tensors_) {
            if (t.offset % alignment != 0) throw Failure("unaligned tensor offset: " + t.name);
            const auto end = checked_add(t.offset, t.bytes);
            if (end > file_size_ - data_start_) throw Failure("tensor exceeds file boundary: " + t.name);
            ranges.emplace_back(t.offset, end);
        }
        std::sort(ranges.begin(), ranges.end());
        std::uint64_t end = 0;
        for (const auto & range : ranges) {
            if (range.first < end) throw Failure("overlapping GGUF tensor ranges");
            end = range.second;
        }
        if (checked_add(data_start_, round_up(end, alignment)) != file_size_)
            throw Failure("GGUF tensor region size mismatch (extra/truncated bytes)");
    }
};

std::string require_string(const Reader & r, const std::string & key) {
    const auto it = r.metadata().find(key);
    if (it == r.metadata().end() || it->second.type != 8) throw Failure("missing GGUF string key: " + key);
    return it->second.text;
}
std::uint64_t require_uint(const Reader & r, const std::string & key) {
    const auto it = r.metadata().find(key);
    if (it == r.metadata().end() || (it->second.type != 4 && it->second.type != 10))
        throw Failure("missing GGUF integer key: " + key);
    return it->second.number;
}
void check_identity(const Reader & reader, const Baseline & baseline) {
    if (require_string(reader, "general.architecture") != EXPECTED_ARCH) throw Failure("unsupported model architecture (must be S1-B1 staging)");
    if (require_string(reader, "loraai.omni.component") != "talker") throw Failure("not a Talker GGUF");
    if (require_string(reader, "loraai.omni.stage") != "S1-B1") throw Failure("unexpected conversion stage");
    const auto it = reader.metadata().find("loraai.omni.inference_ready");
    if (it == reader.metadata().end() || it->second.type != 7 || it->second.flag)
        throw Failure("inference_ready must be GGUF bool false (staging only)");
    if (require_uint(reader, "loraai.omni.expert_count") != 128 || require_uint(reader,"loraai.omni.layer_count") != 20)
        throw Failure("unexpected Talker expert/layer count");
    if (require_string(reader, "loraai.omni.index_sha256") != baseline.source_index_sha)
        throw Failure("source index SHA mismatch");
    if (require_string(reader, "loraai.omni.config_sha256") != baseline.config_sha)
        throw Failure("source config SHA mismatch");
    if (require_string(reader, "loraai.omni.llama_commit") != baseline.llama_commit)
        throw Failure("llama source commit mismatch");
}

std::uint64_t parse_u64(const std::string & x) {
    if (x.empty() || x.find_first_not_of("0123456789") != std::string::npos) throw Failure("non-decimal integer in baseline");
    std::size_t used = 0;
    auto v = std::stoull(x, &used);
    if (used != x.size()) throw Failure("invalid integer in baseline");
    return v;
}
std::vector<std::string> split(const std::string & s, char c) {
    std::vector<std::string> out;
    std::istringstream stream(s);
    std::string part;
    while (std::getline(stream, part, c)) out.push_back(part);
    return out;
}
Baseline load_baseline(const fs::path & file) {
    std::ifstream in(file);
    if (!in) throw Failure("missing layout baseline: " + file.string());
    Baseline b;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            const auto p = line.find('=');
            if (p != std::string::npos) {
                const auto key = line.substr(2, p - 2), value = line.substr(p+1);
                if (key == "source_index_sha256") b.source_index_sha = value;
                if (key == "config_sha256") b.config_sha = value;
                if (key == "llama_commit") b.llama_commit = value;
            }
            continue;
        }
        const auto fields = split(line, '\t');
        if (fields.size() != 4) throw Failure("bad layout baseline TSV row");
        Layout l;
        l.offset = parse_u64(fields[1]);
        l.bytes = parse_u64(fields[2]);
        for (const auto & dim : split(fields[3], ',')) l.dims.push_back(parse_u64(dim));
        if (!b.expected.emplace(fields[0], std::move(l)).second) throw Failure("duplicate name in baseline");
    }
    if (b.source_index_sha.size() != 64 || b.config_sha.size() != 64 || b.llama_commit.size() != 40)
        throw Failure("missing/invalid provenance in baseline");
    if (b.expected.size() != EXPECTED_COUNT) throw Failure("baseline must contain exactly 417 tensors");
    return b;
}
void validate(const Reader & r, const Baseline & b) {
    check_identity(r, b);
    if (r.tensors().size() != EXPECTED_COUNT || r.payload_bytes() != EXPECTED_PAYLOAD)
        throw Failure("unexpected number/total bytes of Talker tensors");
    std::size_t packed_count = 0;
    for (const auto & t : r.tensors()) {
        const auto it = b.expected.find(t.name);
        if (it == b.expected.end()) throw Failure("unrecognized tensor: " + t.name);
        if (t.offset != it->second.offset || t.bytes != it->second.bytes || t.dims != it->second.dims)
            throw Failure("tensor shape/offset/byte mismatch: " + t.name);
        if (t.name.find(".mlp.experts_packed.") != std::string::npos) {
            if (t.dims.size() != 3 || t.dims[2] != 128)
                throw Failure("invalid packed experts axis: " + t.name);
            ++packed_count;
        }
        if (t.name.find(".mlp.experts.") != std::string::npos)
            throw Failure("unpacked expert unexpectedly present: " + t.name);
    }
    if (packed_count != EXPECTED_GROUPS) throw Failure("expected exactly 60 packed expert tensors");
    for (const auto & e : b.expected) r.get_tensor(e.first);
}

std::string hex_preview(const std::vector<std::uint8_t> & bytes) {
    std::ostringstream s;
    for (auto b : bytes) s << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(b);
    return s.str();
}
struct Probe {
    std::string name;
    int expert = 0;
    std::uint64_t slice_bytes = 0;
    std::uint64_t absolute_offset = 0;
    std::string first_32_byte_hex;
};
Probe probe_expert(Reader & r, int layer, int expert, const std::string & proj) {
    if (layer < 0 || layer >= 20 || expert < 0 || expert >= 128 ||
        (proj != "gate_proj" && proj != "up_proj" && proj != "down_proj")) throw Failure("probe argument outside Talker expert range");
    Probe p;
    p.expert = expert;
    p.name = "talker.model.layers." + std::to_string(layer) + ".mlp.experts_packed." + proj + ".weight";
    const auto & t = r.get_tensor(p.name);
    if (t.dims.size() != 3 || t.dims[2] != 128) throw Failure("probe tensor expert dimension not 128");
    p.slice_bytes = checked_mul(checked_mul(t.dims[0], t.dims[1]), 2);
    if (checked_mul(p.slice_bytes, 128) != t.bytes) throw Failure("probe cannot map tensor expert slice");
    const auto slice_start = checked_add(t.offset, checked_mul(static_cast<std::uint64_t>(expert), p.slice_bytes));
    p.absolute_offset = checked_add(r.data_start(), slice_start);
    const auto sample = r.read_region(slice_start, static_cast<std::size_t>(std::min<std::uint64_t>(32, p.slice_bytes)));
    p.first_32_byte_hex = hex_preview(sample);
    return p;
}
std::string quote(const std::string & s) {
    std::string out = "\"";
    for (char ch : s) {
        if (ch == '"' || ch == '\\') { out += '\\'; out += ch; }
        else if (ch == '\n') out += "\\n";
        else if (static_cast<unsigned char>(ch) < 0x20) throw Failure("non-printable JSON string");
        else out += ch;
    }
    return out + '"';
}
std::string report(const Reader & r, const std::optional<Probe> & p) {
    std::ostringstream j;
    j << "{\n  \"schema\": \"lora.omni.s1b2.v1\",\n  \"ok\": true,\n"
      << "  \"stage\": \"S1-B2-metadata-only\",\n  \"inference_ready\": false,\n"
      << "  \"tensors\": " << r.tensors().size() << ",\n"
      << "  \"packed_groups\": " << EXPECTED_GROUPS << ",\n"
      << "  \"payload_bytes\": " << r.payload_bytes() << ",\n"
      << "  \"file_bytes\": " << r.file_size() << ",\n"
      << "  \"data_start\": " << r.data_start();
    if (p) {
        j << ",\n  \"probe\": {\n    \"tensor\": " << quote(p->name)
          << ",\n    \"expert\": " << p->expert
          << ",\n    \"slice_bytes\": " << p->slice_bytes
          << ",\n    \"absolute_offset\": " << p->absolute_offset
          << ",\n    \"first_32_byte_hex\": " << quote(p->first_32_byte_hex) << "\n  }";
    }
    j << "\n}\n";
    return j.str();
}

int run(int argc, char ** argv) {
    fs::path model = "omni-experimental/talker-experts-packed-bf16-staging.gguf";
    fs::path baseline, report_path;
    std::optional<std::tuple<int,int,std::string>> probe;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            std::cout << "omni_talker_probe --baseline layout.tsv [--model packed.gguf] [--probe LAYER EXPERT PROJECTION] [--report file.json]\n";
            return 0;
        } else if (arg == "--model" && i+1 < argc) model = argv[++i];
        else if (arg == "--baseline" && i+1 < argc) baseline = argv[++i];
        else if (arg == "--report" && i+1 < argc) report_path = argv[++i];
        else if (arg == "--probe" && i+3 < argc) {
            const int layer = std::stoi(argv[++i]);
            const int expert = std::stoi(argv[++i]);
            const std::string projection = argv[++i];
            probe = std::make_tuple(layer, expert, projection);
        } else throw Failure("unknown/missing CLI argument: " + arg);
    }
    if (baseline.empty()) throw Failure("--baseline is required; cannot validate weights without pinned layout");
    Reader r(model);
    const auto b = load_baseline(baseline);
    validate(r, b);
    std::optional<Probe> probed;
    if (probe) probed = probe_expert(r, std::get<0>(*probe),std::get<1>(*probe),std::get<2>(*probe));
    const auto result = report(r, probed);
    if (!report_path.empty()) {
        if (fs::absolute(report_path).lexically_normal() == fs::absolute(model).lexically_normal())
            throw Failure("report path cannot equal model path");
        std::ofstream out(report_path);
        if (!out) throw Failure("cannot write report: " + report_path.string());
        out << result;
        if (!out.good()) throw Failure("report write failed");
    }
    std::cout << "S1-B2 VERIFIED: 417 Talker BF16 tensors; 60 packed expert groups; metadata and offsets match pinned S0-B baseline.\n"
              << "STAGING ONLY: No GGML graph, no Thinker bridge, no audio inference.\n"
              << result;
    return 0;
}
} // namespace omni
} // internal linkage
