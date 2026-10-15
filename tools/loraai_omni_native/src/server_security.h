#pragma once
#include <arpa/inet.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace lora_omni {
inline bool loopback_address(const std::string & host) {
    in_addr v4{};
    if (inet_pton(AF_INET, host.c_str(), &v4) == 1) return (ntohl(v4.s_addr) >> 24) == 127;
    in6_addr v6{};
    if (inet_pton(AF_INET6, host.c_str(), &v6) == 1) {
        if (IN6_IS_ADDR_LOOPBACK(&v6)) return true;
        return IN6_IS_ADDR_V4MAPPED(&v6) && v6.s6_addr[12] == 127;
    }
    return false;
}

inline void validate_server_binding(std::string & host, const std::vector<std::string> & keys) {
    // Pin localhost to a literal loopback address instead of trusting DNS.
    if (host == "localhost") host = "127.0.0.1";
    for (const auto & key : keys) {
        if (key.find_first_not_of(" \t\r\n") == std::string::npos)
            throw std::runtime_error("Omni API keys must not be empty");
    }
    if (!loopback_address(host) && keys.empty())
        throw std::runtime_error("Omni refuses a non-loopback address without --api-key or --api-key-file");
}
}
