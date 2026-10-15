// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andrei Tolkachev
// Derived from the user-provided, BF16-verified LoraAI kernels. See provenance.json.
#pragma once
// S1-B5: small, reusable Qwen3-Omni Talker attention math contract.
// CPU scalar RMSNorm, multimodal rotary, GQA, incremental KV cache;
// weight projection graphs are executed separately through GGML.
// NOT complete decoder-layer execution or finished Talker inference.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace omni_attn {
constexpr int HIDDEN = 1024;
constexpr int N_HEAD = 16;
constexpr int N_KV_HEAD = 2;
constexpr int HEAD_DIM = 128;
constexpr int N_GROUP = N_HEAD/N_KV_HEAD;
constexpr double EPS = 1e-6;
constexpr double THETA = 1000000.0;
constexpr std::array<int,3> MROPE_SECTION = {24,20,20};

inline void guard(bool ok,const char* why) { if(!ok) throw std::runtime_error(why); }
inline void finite(const std::vector<float>& x) {
    for(float y:x) guard(std::isfinite(y),"non-finite attention vector");
}
inline std::vector<float> deterministic_input(int position) {
    guard(position>=0 && position<64,"invalid synthetic input position");
    std::vector<float> x(HIDDEN);
    for(int i=0;i<HIDDEN;++i) {
        const double s=std::sin((i+1)*0.023+position*0.19)*0.03125;
        const double c=std::cos((i+3)*0.007-position*0.13)*0.015625;
        x[i]=static_cast<float>(s+c);
    }
    return x;
}
// RMS norm across only the specified last dimension; Q/K are normalized PER HEAD.
inline void rms_norm_inplace(float* x,const float* gain,int n,double eps=EPS) {
    guard(x!=nullptr && gain!=nullptr && n>0,"invalid RMSNorm buffers");
    double acc=0;
    for(int i=0;i<n;++i) {
        guard(std::isfinite(x[i]) && std::isfinite(gain[i]),"non-finite RMSNorm input");
        acc+=static_cast<double>(x[i])*x[i];
    }
    const double scale=1.0/std::sqrt(acc/n+eps);
    for(int i=0;i<n;++i) x[i]=static_cast<float>(static_cast<double>(x[i])*scale*gain[i]);
}
inline void norm_heads(std::vector<float>& vec,const std::vector<float>& gain,int heads) {
    guard(heads>0 && int(vec.size())==heads*HEAD_DIM && gain.size()==HEAD_DIM,"invalid Q/K norm shape");
    for(int h=0;h<heads;++h) rms_norm_inplace(vec.data()+h*HEAD_DIM,gain.data(),HEAD_DIM);
}
// Interleaved M-RoPE refers to *frequency-axis ownership*, not adjacent-pair rotation.
// Qwen3-VL uses rotate_half: component i pairs with i + HEAD_DIM/2.
inline int mrope_axis(int freq_index) {
    guard(freq_index>=0 && freq_index<HEAD_DIM/2,"invalid M-RoPE frequency index");
    if(freq_index%3==1 && freq_index<3*MROPE_SECTION[1]) return 1;
    if(freq_index%3==2 && freq_index<3*MROPE_SECTION[2]) return 2;
    return 0;
}
inline void apply_mrope(float* x,const std::array<int,3>& position,double theta=THETA) {
    guard(x!=nullptr && std::isfinite(theta) && theta>1.0,"invalid M-RoPE settings");
    for(int p:position) guard(p>=0 && p<65536,"invalid M-RoPE position");
    constexpr int HALF=HEAD_DIM/2;
    for(int i=0;i<HALF;++i) {
        const double angle=static_cast<double>(position[mrope_axis(i)])*
            std::pow(theta,-2.0*i/HEAD_DIM);
        const double c=std::cos(angle),s=std::sin(angle);
        const double first=x[i],second=x[i+HALF];
        x[i]=static_cast<float>(first*c-second*s);
        x[i+HALF]=static_cast<float>(second*c+first*s);
    }
}
inline void rotate_heads(std::vector<float>& vec,int heads,const std::array<int,3>& position) {
    guard(heads>0 && int(vec.size())==heads*HEAD_DIM,"invalid M-RoPE vector shape");
    for(int h=0;h<heads;++h) apply_mrope(vec.data()+h*HEAD_DIM,position);
    finite(vec);
}
class KVCache {
    std::vector<float> keys_,values_;
    int tokens_=0;
public:
    explicit KVCache(int capacity=64):capacity_(capacity) {
        guard(capacity>0 && capacity<=4096,"invalid KV cache capacity");
    }
    const int capacity_;
    int tokens() const {return tokens_;}
    void clear() {keys_.clear();values_.clear();tokens_=0;}
    void append(int token_index,const std::vector<float>& k,const std::vector<float>& v) {
        guard(token_index==tokens_ && tokens_<capacity_,"KV token index or capacity mismatch");
        guard(k.size()==N_KV_HEAD*HEAD_DIM && v.size()==N_KV_HEAD*HEAD_DIM,"invalid KV vector shapes");
        finite(k);finite(v);
        keys_.insert(keys_.end(),k.begin(),k.end());
        values_.insert(values_.end(),v.begin(),v.end());
        ++tokens_;
    }
    const float* key(int token,int head) const {
        guard(token>=0 && token<tokens_ && head>=0 && head<N_KV_HEAD,"invalid KV key index");
        return keys_.data()+(token*N_KV_HEAD+head)*HEAD_DIM;
    }
    const float* value(int token,int head) const {
        guard(token>=0 && token<tokens_ && head>=0 && head<N_KV_HEAD,"invalid KV value index");
        return values_.data()+(token*N_KV_HEAD+head)*HEAD_DIM;
    }
};
inline std::vector<float> attend(const std::vector<float>& q,const KVCache& cache) {
    guard(q.size()==N_HEAD*HEAD_DIM && cache.tokens()>0,"invalid causal attention input");
    finite(q);
    std::vector<float> y(N_HEAD*HEAD_DIM,0.f);
    for(int h=0;h<N_HEAD;++h) {
        const int kvh=h/N_GROUP;
        const float* query=q.data()+h*HEAD_DIM;
        std::vector<double> logits(cache.tokens());
        double top=-std::numeric_limits<double>::infinity();
        for(int t=0;t<cache.tokens();++t) {
            const float* k=cache.key(t,kvh);
            double sum=0;
            for(int j=0;j<HEAD_DIM;++j) sum+=static_cast<double>(query[j])*k[j];
            logits[t]=sum/std::sqrt(double(HEAD_DIM));
            top=std::max(top,logits[t]);
        }
        double normalizer=0;
        for(double& z:logits) {z=std::exp(z-top);normalizer+=z;}
        guard(normalizer>0 && std::isfinite(normalizer),"invalid attention softmax");
        for(int t=0;t<cache.tokens();++t) {
            const double p=logits[t]/normalizer;
            const float* v=cache.value(t,kvh);
            for(int j=0;j<HEAD_DIM;++j)
                y[h*HEAD_DIM+j]+=static_cast<float>(p*v[j]);
        }
    }
    finite(y);
    return y;
}
} // namespace omni_attn
