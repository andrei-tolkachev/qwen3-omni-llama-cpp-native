#include "backend.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>
namespace lora_omni {
struct DeviceDecoderTest {
    static void run(const std::string & path) {
        DeviceDecoder growing(path, 1300, true), reference(path, 1300, true);
        reference.grow(1300);
        for (int step = 0; step < 1300; ++step) {
            Vector input(1024);
            for (int j = 0; j < 1024; ++j) input[j] = std::sin(float(step * 31 + j) * .013f) * .1f;
            const auto a = growing.step(input), b = reference.step(input);
            assert(a.size() == b.size());
            for (size_t j = 0; j < a.size(); ++j) assert(std::isfinite(a[j]) && std::abs(a[j] - b[j]) < 1e-5f);
            if (step == 512 || step == 1024 || step == 1299) {
                for (int i = 0; i < growing.layers; ++i) {
                    for (bool key : {false, true}) {
                        auto * x = (key ? growing.keys : growing.values)[i];
                        auto * y = (key ? reference.keys : reference.values)[i];
                        Vector actual(size_t(step + 1) * 128 * growing.kv_heads), expected(actual.size());
                        ggml_backend_tensor_get(x, actual.data(), 0, actual.size() * sizeof(float));
                        ggml_backend_tensor_get(y, expected.data(), 0, expected.size() * sizeof(float));
                        assert(actual == expected);
                    }
                }
                std::fprintf(stdout, "PASS: position %d, all KV bytes preserved, output equals preallocated reference\n", step);
                std::fflush(stdout);
            }
        }
        growing.rewind(512); reference.rewind(512);
        for(int i=512;i<520;++i) {
            Vector value(1024,.03f);
            const auto a=growing.step(value), b=reference.step(value);
            assert(a==b);
        }
        const auto expected=growing.step(Vector(1024,.07f));
        growing.rewind(520);
        const auto repeated=growing.step(Vector(1024,.07f));
        assert(expected==repeated);
        try{growing.rewind(522);assert(false);}catch(const std::runtime_error&){}
        std::puts("PASS: rewound suffix reproduces identical hidden states; future position rejected");

    }
};
}
int main(int argc, char ** argv) { assert(argc == 2); lora_omni::DeviceCacheScope scope; lora_omni::DeviceDecoderTest::run(argv[1]); }
