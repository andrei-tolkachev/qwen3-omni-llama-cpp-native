#include "backend.h"
#include "omni.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>

namespace fs = std::filesystem;
using namespace lora_omni;
static Vector read_rows(const std::string & path) {
    const auto bytes = fs::file_size(path);
    if (!bytes || bytes % 4096 || bytes > 4096 * 4096) throw std::runtime_error("invalid F32 row file");
    Vector value(bytes / 4);
    std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char *>(value.data()), bytes);
    if (!file) throw std::runtime_error("short input read");
    for (float v : value) if (!std::isfinite(v)) throw std::runtime_error("nonfinite input");
    return value;
}
static Vector row(const Vector & value, int index) {
    return Vector(value.begin() + size_t(index) * 1024, value.begin() + size_t(index + 1) * 1024);
}
static void u32(std::ostream & out, uint32_t value) {
    for (int i = 0; i < 4; ++i) out.put(char((value >> (8 * i)) & 255));
}
int main(int argc, char ** argv) {
    try {
        lora_omni::DeviceCacheScope device_cache_scope;
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 == argc) throw std::runtime_error("option requires a value");
            args[argv[i]] = argv[i + 1];
        }
        for (const auto * key : {"--talker", "--predictor", "--prefill", "--trailing", "--pad", "--out"})
            if (!args.count(key)) throw std::runtime_error(std::string("missing ") + key);
        const int threads = args.count("--threads") ? std::stoi(args["--threads"]) : 4;
        const int max_frames = args.count("--max-frames") ? std::stoi(args["--max-frames"]) : 300;
        if (threads < 1 || threads > 32 || max_frames < 1 || max_frames > 512) throw std::runtime_error("invalid limits");
        const auto prefill = read_rows(args["--prefill"]), trailing = read_rows(args["--trailing"]), pad = read_rows(args["--pad"]);
        if (pad.size() != 1024 || prefill.size() / 1024 + max_frames > 4096) throw std::runtime_error("context capacity exceeded");
        const fs::path output(args["--out"]);
        if (fs::exists(output) || fs::exists(output.string() + ".json")) throw std::runtime_error("output already exists");
        Talker talker(args["--talker"], int(prefill.size() / 1024) + max_frames, threads);
        Predictor predictor(args["--predictor"], threads);
        const auto started = std::chrono::steady_clock::now();
        Vector hidden;
        for (int i = 0; i < int(prefill.size() / 1024); ++i) hidden = talker.step(row(prefill, i));
        std::vector<Codes> frames;
        bool eos = false;
        for (int i = 0; i < max_frames; ++i) {
            const int code = select_codec(talker.logits(hidden));
            if (code == 2150) { eos = true; break; }
            auto prediction = predictor.frame(hidden, code, talker.embedding(code));
            frames.push_back(prediction.codes);
            const auto text = size_t(i + 1) * 1024 <= trailing.size() ? row(trailing, i) : pad;
            for (int j = 0; j < 1024; ++j) prediction.feedback[j] += text[j];
            std::cerr << "codec frame " << i << ": " << code << std::endl;
            if (i + 1 < max_frames) hidden = talker.step(prediction.feedback);
        }
        if (frames.empty()) throw std::runtime_error("Talker stopped without audio frames");
        std::ofstream file(output, std::ios::binary);
        file.write("Q3OC", 4); u32(file, 1); u32(file, uint32_t(frames.size())); u32(file, 16);
        for (const auto & frame : frames) for (int code : frame) { file.put(char(code & 255)); file.put(char((code >> 8) & 255)); }
        file.close();
        if (!file) throw std::runtime_error("codec output write failed");
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::ofstream report(output.string() + ".json");
        report << "{\"schema\":\"lora.omni.native.codec.v1\",\"frames\":" << frames.size()
               << ",\"codebooks\":16,\"eos\":" << (eos ? "true" : "false")
               << ",\"finish_reason\":\"" << (eos ? "stop" : "length")
               << "\",\"sampling\":\"greedy\",\"seconds\":" << elapsed << "}\n";
        if (!report) throw std::runtime_error("report write failed");
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
