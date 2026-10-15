// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andrei Tolkachev
// Derived from the user-provided, BF16-verified LoraAI kernels. See provenance.json.
#pragma once
// Native Qwen3-Omni Talker TextConfig: top-6 / 128, normalized softmax routing.
// This is intentionally restricted to a single token, no attention/KV cache.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lora_moe {
constexpr int N_EXPERTS=128;
constexpr int TOP_K=6;
constexpr int HIDDEN=1024;
constexpr int EXPERT_INTERMEDIATE=384;
constexpr int SHARED_INTERMEDIATE=768;

struct Routing {
    std::vector<int> ids;
    std::vector<double> weights;
    std::vector<double> probabilities;
};

inline Routing route(const std::vector<float> & logits) {
    if (logits.size()!=N_EXPERTS) throw std::runtime_error("router logits size != 128");
    float max_logit=-std::numeric_limits<float>::infinity();
    for (const float v : logits) {
        if (!std::isfinite(v)) throw std::runtime_error("nonfinite router logit");
        max_logit=std::max(max_logit,v);
    }
    std::vector<double> p(N_EXPERTS);
    double denom=0;
    for(int i=0;i<N_EXPERTS;++i) {
        p[i]=std::exp(static_cast<double>(logits[i])-max_logit);
        denom+=p[i];
    }
    if (!std::isfinite(denom) || denom<=0) throw std::runtime_error("bad softmax normalization");
    for(auto & v:p) v/=denom;
    std::vector<int> order(N_EXPERTS);
    std::iota(order.begin(),order.end(),0);
    // Deterministic tie break; model inputs are not expected to contain exact ties.
    std::partial_sort(order.begin(),order.begin()+TOP_K,order.end(),[&](int a,int b){
        return p[a]==p[b] ? a<b : p[a]>p[b];
    });
    Routing r;
    r.probabilities=std::move(p);
    double top_sum=0;
    for (int i=0;i<TOP_K;++i) {
        r.ids.push_back(order[i]);
        top_sum+=r.probabilities[order[i]];
    }
    if (!std::isfinite(top_sum)||top_sum<=0) throw std::runtime_error("invalid top-6 sum");
    for (int id:r.ids) r.weights.push_back(r.probabilities[id]/top_sum);
    return r;
}

inline double sigmoid(double x) {
    if (!std::isfinite(x)) throw std::runtime_error("invalid shared expert logit");
    return x>=0 ? 1.0/(1.0+std::exp(-x)) : std::exp(x)/(1.0+std::exp(x));
}

inline std::vector<float> combine(const Routing & routing,
                                  const std::vector<std::vector<float>> & routed,
                                  const std::vector<float> & shared,
                                  double shared_weight) {
    if (routing.ids.size()!=TOP_K || routing.weights.size()!=TOP_K || routed.size()!=TOP_K ||
        shared.size()!=HIDDEN || !std::isfinite(shared_weight) || shared_weight<0 || shared_weight>1)
        throw std::runtime_error("invalid MoE aggregation shape or gate");
    std::vector<float> y(HIDDEN);
    for (int i=0;i<HIDDEN;++i) {
        double sum=static_cast<double>(shared[i])*shared_weight;
        for(int j=0;j<TOP_K;++j) {
            if (routed[j].size()!=HIDDEN || !std::isfinite(routed[j][i]))
                throw std::runtime_error("nonfinite or malformed expert output");
            sum += routing.weights[j]*routed[j][i];
        }
        if (!std::isfinite(sum)) throw std::runtime_error("nonfinite MoE aggregate");
        y[i]=static_cast<float>(sum);
    }
    return y;
}
} // namespace lora_moe
