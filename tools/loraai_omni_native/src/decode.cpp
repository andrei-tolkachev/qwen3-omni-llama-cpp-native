#include "backend.h"
#include "voice.h"
#include "codec_stream.h"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc, char ** argv) {
    try {
        lora_omni::DeviceCacheScope device_cache_scope;
        if (argc != 4 && argc != 5) throw std::runtime_error("usage: omni-decode CODE2WAV.gguf FRAMES.q3oc OUTPUT.wav [CHUNK_FRAMES]");
        std::ifstream input(argv[2], std::ios::binary);
        const std::string data((std::istreambuf_iterator<char>(input)), {});
        auto u32 = [&](size_t offset) { uint32_t n = 0; for (int i = 0; i < 4; ++i) n |= uint32_t(uint8_t(data.at(offset + i))) << (8 * i); return n; };
        if (data.size() < 16 || data.substr(0, 4) != "Q3OC" || u32(4) != 1 || u32(12) != 16 ||
            !u32(8) || u32(8) > 512 || data.size() != 16 + u32(8) * 32) throw std::runtime_error("invalid Q3OC");
        if (std::filesystem::exists(argv[3]) || std::filesystem::exists(std::string(argv[3]) + ".f32"))
            throw std::runtime_error("output already exists");
        std::vector<lora_omni::Codes> codes(u32(8));
        for (size_t i = 0; i < codes.size(); ++i) for (int j = 0; j < 16; ++j) {
            const auto offset = 16 + i * 32 + j * 2;
            codes[i][j] = uint8_t(data[offset]) | (uint16_t(uint8_t(data[offset + 1])) << 8);
        }
        lora_omni::Vector pcm;
        if (argc == 5) {
            const int chunk = std::stoi(argv[4]);
            if (chunk < 1 || chunk > 16) throw std::runtime_error("invalid streaming chunk size");
            lora_omni::CodecStream decoder(argv[1]);
            for (size_t i = 0; i < codes.size(); i += chunk) {
                const std::vector<lora_omni::Codes> part(codes.begin() + i, codes.begin() + std::min(codes.size(), i + chunk));
                auto samples = decoder.push(part);
                pcm.insert(pcm.end(), samples.begin(), samples.end());
            }
        } else pcm = lora_omni::decode_audio(argv[1], codes, 4);
        const auto wav = lora_omni::wav_bytes(pcm);
        std::ofstream output(argv[3], std::ios::binary);
        output.write(wav.data(), wav.size()); output.close();
        if (!output) throw std::runtime_error("WAV write failed");
        std::ofstream floats(std::string(argv[3]) + ".f32", std::ios::binary);
        floats.write(reinterpret_cast<const char *>(pcm.data()), pcm.size() * 4); floats.close();
        if (!floats) throw std::runtime_error("PCM write failed");
        std::cout << "frames=" << codes.size() << " samples=" << pcm.size() << '\n';
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
