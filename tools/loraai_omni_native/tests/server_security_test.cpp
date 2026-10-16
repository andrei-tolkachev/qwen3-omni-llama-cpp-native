#include "server_security.h"
#include "common.h"
#include "server-http.h"
#include <cpp-httplib/httplib.h>
#include <cassert>
#include <chrono>
#include <thread>

int main() {
    for (auto host : {"localhost", "127.0.0.1", "127.2.3.4", "::1", "::ffff:127.0.0.1"}) {
        std::string address(host);
        lora_omni::validate_server_binding(address, {});
        assert(lora_omni::loopback_address(address));
    }
    for (auto host : {"0.0.0.0", "::", "192.0.2.1", "example.test", "127.0.0.1.example.test", ""}) {
        std::string address(host);
        bool refused = false;
        try { lora_omni::validate_server_binding(address, {}); }
        catch (const std::runtime_error &) { refused = true; }
        assert(refused);
        lora_omni::validate_server_binding(address, {"test-only-key"});
    }
    for (auto key : {"", " \t"}) {
        std::string address("0.0.0.0");
        bool refused = false;
        try { lora_omni::validate_server_binding(address, {key}); }
        catch (const std::runtime_error &) { refused = true; }
        assert(refused);
    }
    common_params params;
    params.hostname = "127.0.0.1";
    params.port = 0;
    params.api_keys = {"test-only-key"};
    server_http_context http;
    assert(http.init(params));
    int called = 0;
    const std::vector<std::string> paths = {"/v1/omni/voice", "/v1/omni/cancel", "/v1/omni/duplex",
        "/v1/omni/duplex/input", "/v1/omni/duplex/cancel", "/v1/omni/duplex/close", "/v1/omni/duplex/ack",
        "/v1/omni/duplex/context", "/v1/omni/duplex/query", "/v1/omni/duplex/interrupt-input"};
    for (const auto & path : paths) {
        http.post(path, [&](const server_http_req &) {
            ++called;
            auto response = std::make_unique<server_http_res>();
            response->data = "{}";
            return response;
        });
    }
    http.is_ready = true;
    assert(http.start());
    httplib::Client client("127.0.0.1", http.port);
    client.set_connection_timeout(5);
    for (const auto & path : paths) {
        auto missing = client.Post(path, "{}", "application/json");
        assert(missing && missing->status == 401);
        auto wrong = client.Post(path, {{"Authorization", "Bearer wrong-test-key"}}, "{}", "application/json");
        assert(wrong && wrong->status == 401);
        auto good = client.Post(path, {{"Authorization", "Bearer test-only-key"}}, "{}", "application/json");
        assert(good && good->status == 200);
    }
    http.stop();
    http.thread.join();
    assert(called == int(paths.size()));
}
