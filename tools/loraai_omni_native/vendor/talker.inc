// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andrei Tolkachev
// Derived from the user-provided, BF16-verified LoraAI kernels. See provenance.json.
// LoraAI Qwen3-Omni experimental Talker S1-B14K: 20 consecutive decoder layers, real BF16 weights.
// Reuses the numerically verified S1-B6 layer math; changes the input handoff.
// Attention & MoE matrix products use GGML CPU; RMSNorm, M-RoPE, causal GQA,
// Top-6 routing and KV cache remain scalar C++ validation scaffolding.
// Original B14F native prefill input to layer 0; every subsequent layer consumes previous output. 24-row original B14F prefill replayed, then one authentic B14J frame0 codec-feedback row in the same KV cache.
// Prefill REPLAY + codec-feedback continuation from B14J; NOT multiple codec frames, speech or server inference.
#include "expert.inc"
#include "router.h"
#include "attention.h"
#include <array>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <memory>

namespace {
namespace att=omni_attn;
std::vector<float> weight_b6(omni::Reader& r,const std::string&name,
                             std::uint64_t d0,std::uint64_t d1=0){
    const auto &t=r.get_tensor(name);
    auto dims=d1?std::vector<std::uint64_t>{d0,d1}:std::vector<std::uint64_t>{d0};
    if(t.dims!=dims||t.type!=omni::GGUF_BF16||t.bytes!=2*d0*(d1?d1:1))
        throw omni::Failure("S1-B6 bad shape/type: "+name);
    return to_f32(r.read_region(t.offset,static_cast<std::size_t>(t.bytes)));
}
std::vector<float> mul_b6(const std::vector<float>&w,const std::vector<float>&x,
                          int d_in,int d_out,int threads) {
    if(w.size()!=static_cast<std::size_t>(d_in)*d_out||x.size()!=static_cast<std::size_t>(d_in))
        throw omni::Failure("S1-B6 invalid matmul geometry");
    ggml_init_params p{};
    p.mem_size=w.size()*sizeof(float)+16*1024*1024;
    p.no_alloc=false;
    auto *ctx=ggml_init(p);
    if(!ctx)throw omni::Failure("S1-B6 ggml_init failed");
    struct Guard{ggml_context*p;~Guard(){ggml_free(p);}}guard{ctx};
    auto *wt=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,d_in,d_out);
    auto *vt=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,d_in,1);
    if(!wt||!vt)throw omni::Failure("S1-B6 GGML matmul alloc failed");
    std::memcpy(ggml_get_data(wt),w.data(),w.size()*sizeof(float));
    std::memcpy(ggml_get_data(vt),x.data(),x.size()*sizeof(float));
    auto *out=ggml_mul_mat(ctx,wt,vt);
    auto *graph=ggml_new_graph(ctx);
    ggml_build_forward_expand(graph,out);
    if(ggml_graph_compute_with_ctx(ctx,graph,threads)!=GGML_STATUS_SUCCESS)
        throw omni::Failure("S1-B6 GGML matmul error");
    std::vector<float> y(d_out);
    std::memcpy(y.data(),ggml_get_data(out),y.size()*sizeof(float));
    att::finite(y);
    return y;
}
void plus_b6(std::vector<float>&out,const std::vector<float>&rhs){
    if(out.size()!=rhs.size())throw omni::Failure("S1-B6 residual geometry mismatch");
    for(std::size_t i=0;i<out.size();++i)out[i]+=rhs[i];
    att::finite(out);
}
double maximum_b6(const std::vector<float>&v){
    double m=0;for(float z:v)m=std::max(m,double(std::abs(z)));return m;
}
struct LayerB6 {
    int idx,threads;
    std::vector<float> input_gain,post_gain,q_gain,k_gain;
    std::vector<float> qw,kw,vw,ow,router,shared_gate;
    Weights shared;
    omni::Reader& reader;
    att::KVCache cache;
    LayerB6(omni::Reader &r,int layer,int nsteps,int nthreads):
      idx(layer),threads(nthreads),reader(r),cache(nsteps) {
        const std::string root="talker.model.layers."+std::to_string(idx)+".";
        input_gain=weight_b6(r,root+"input_layernorm.weight",1024);
        post_gain=weight_b6(r,root+"post_attention_layernorm.weight",1024);
        q_gain=weight_b6(r,root+"self_attn.q_norm.weight",128);
        k_gain=weight_b6(r,root+"self_attn.k_norm.weight",128);
        qw=weight_b6(r,root+"self_attn.q_proj.weight",1024,2048);
        kw=weight_b6(r,root+"self_attn.k_proj.weight",1024,256);
        vw=weight_b6(r,root+"self_attn.v_proj.weight",1024,256);
        ow=weight_b6(r,root+"self_attn.o_proj.weight",2048,1024);
        router=weight_b6(r,root+"mlp.gate.weight",1024,128);
        shared_gate=weight_b6(r,root+"mlp.shared_expert_gate.weight",1024,1);
        shared.hidden=1024;shared.intermediate=768;
        shared.gate=weight_b6(r,root+"mlp.shared_expert.gate_proj.weight",1024,768);
        shared.up=weight_b6(r,root+"mlp.shared_expert.up_proj.weight",1024,768);
        shared.down=weight_b6(r,root+"mlp.shared_expert.down_proj.weight",768,1024);
    }
    struct Step {
        std::vector<float> attn,post_attn,moe,output;
        lora_moe::Routing routing;
        double shared_gate_probability=0,shared_signal=0;
    };
    Step eval(int pos,const std::vector<float>& x){
        if(x.size()!=1024)throw omni::Failure("S1-B7 activation width must be 1024");
        att::finite(x); // explicit upstream activation, not a fresh input per layer
        auto norm=x;
        att::rms_norm_inplace(norm.data(),input_gain.data(),1024);
        auto q=mul_b6(qw,norm,1024,2048,threads);
        auto k=mul_b6(kw,norm,1024,256,threads);
        auto v=mul_b6(vw,norm,1024,256,threads);
        att::norm_heads(q,q_gain,16);
        att::norm_heads(k,k_gain,2);
        const std::array<int,3> axes={pos,pos,pos};
        att::rotate_heads(q,16,axes);
        att::rotate_heads(k,2,axes);
        cache.append(pos,k,v);
        auto context=att::attend(q,cache);
        Step o;
        o.attn=mul_b6(ow,context,2048,1024,threads);
        o.post_attn=x;
        plus_b6(o.post_attn,o.attn); // residual #1
        auto moe_x=o.post_attn;
        att::rms_norm_inplace(moe_x.data(),post_gain.data(),1024);
        auto logits=mul_b6(router,moe_x,1024,128,threads);
        o.routing=lora_moe::route(logits);
        std::vector<std::vector<float>> routed;
        routed.reserve(6);
        for(int id:o.routing.ids)routed.push_back(compute(load_one(reader,idx,id),moe_x,threads));
        auto shared_out=compute(shared,moe_x,threads);
        double shared_logit=0;
        for(int i=0;i<1024;++i)shared_logit+=double(shared_gate[i])*moe_x[i];
        o.shared_gate_probability=lora_moe::sigmoid(shared_logit);
        o.shared_signal=maximum_b6(shared_out)*o.shared_gate_probability;
        o.moe=lora_moe::combine(o.routing,routed,shared_out,o.shared_gate_probability);
        o.output=o.post_attn;
        plus_b6(o.output,o.moe); // residual #2
        if(maximum_b6(o.attn)<1e-9 || maximum_b6(o.moe)<1e-10 || maximum_b6(o.output)<1e-9||
           o.shared_signal<1e-11)
            throw omni::Failure("S1-B6 degenerate layer signal");
        return o;
    }
};

}
