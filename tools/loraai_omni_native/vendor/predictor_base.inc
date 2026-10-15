// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andrei Tolkachev
// Derived from the user-provided, BF16-verified LoraAI kernels. See provenance.json.
// Exact accepted S1-B10 C++ math; main renamed only. Original unchanged.
// S1-B10: Qwen3-Omni native Code Predictor: BF16 staging GGUF, 5-layer prefill,
// first residual codec head (group 1). No Transformers, CUDA, or external TTS.
// Output is a numeric *microtest* with synthetic upstream conditioning.
// NOT a full 15-step RVQ sequence, not Code2Wav, not speech inference.
#include "expert.inc"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <set>
#include <vector>

namespace {
namespace fs = std::filesystem;
constexpr int H=1024, MID=3072, QHEADS=16, KVHEADS=8, HEAD=128, QDIM=2048, KVDIM=1024;
constexpr int CODEVOCAB=2048, LAYERS=5, GROUPS=16;
constexpr double EPS=1e-6, THETA=1000000.0;
struct CPArgs {
    fs::path model="omni-experimental/code-predictor-staging-bf16.gguf";
    fs::path baseline,hidden,embedding,out;
    int steps=2,threads=2,codec0=-1;
    bool self_test=false;
};
CPArgs cp_parse(int argc,char**argv){
    CPArgs a;
    for(int i=1;i<argc;++i){
        const std::string k=argv[i];
        if(k=="--self-test")a.self_test=true;
        else if(k=="--model"&&i+1<argc)a.model=argv[++i];
        else if(k=="--baseline"&&i+1<argc)a.baseline=argv[++i];
        else if(k=="--talker-hidden"&&i+1<argc)a.hidden=argv[++i];
        else if(k=="--codec0-embedding"&&i+1<argc)a.embedding=argv[++i];
        else if(k=="--output-prefix"&&i+1<argc)a.out=argv[++i];
        else if(k=="--steps"&&i+1<argc)a.steps=std::stoi(argv[++i]);
        else if(k=="--threads"&&i+1<argc)a.threads=std::stoi(argv[++i]);
        else if(k=="--codec0-id"&&i+1<argc)a.codec0=std::stoi(argv[++i]);
        else if(k=="--help"){
            std::cout<<"omni_code_predictor_ggml --model CP_STAGING_GGUF --baseline TSV "
                "--talker-hidden F32 --codec0-embedding F32 --codec0-id ID "
                "--output-prefix PATH [--steps 2|3] [--threads 2]\n"
                "Five-layer predictor FIRST residual codec group on synthetic upstream, NOT speech.\n";
            std::exit(0);
        }else throw omni::Failure("S1-B10 unknown or incomplete option: "+k);
    }
    if(a.steps<2||a.steps>3||a.threads<1||a.threads>16||
       (!a.self_test && (a.baseline.empty()||a.hidden.empty()||a.embedding.empty()||
                         a.out.empty()||a.codec0<0||a.codec0>=3072)))
        throw omni::Failure("S1-B10 missing or invalid args");
    return a;
}

void cp_verify_layout(omni::Reader& r,const fs::path& layout){
    std::ifstream f(layout);
    if(!f)throw omni::Failure("S1-B10 pinned layout not found");
    std::string line;
    std::size_t count=0;
    std::set<std::string> names;
    while(std::getline(f,line)){
        if(line.empty()||line[0]=='#')continue;
        std::istringstream ss(line);
        std::string name,bytes,dimensions;
        if(!std::getline(ss,name,'\t')||!std::getline(ss,bytes,'\t')||
           !std::getline(ss,dimensions,'\t'))
            throw omni::Failure("S1-B10 malformed layout entry");
        if(!names.insert(name).second)throw omni::Failure("S1-B10 duplicate pinned layout entry");
        const auto&t=r.get_tensor(name);
        if(t.type!=omni::GGUF_BF16||t.bytes!=std::stoull(bytes))
            throw omni::Failure("S1-B10 BF16 size/type mismatch: "+name);
        std::ostringstream current;
        for(std::size_t i=0;i<t.dims.size();++i){
            if(i)current<<",";
            current<<t.dims[i];
        }
        if(current.str()!=dimensions)
            throw omni::Failure("S1-B10 shape mismatch: "+name);
        ++count;
    }
    if(count!=86||r.tensors().size()!=86)
        throw omni::Failure("S1-B10 incomplete Code Predictor tensor inventory");
    const auto&kv=r.metadata();
    auto need_str=[&](const std::string&key,const std::string&wanted){
        auto it=kv.find(key);
        if(it==kv.end()||it->second.type!=8||it->second.text!=wanted)
            throw omni::Failure("S1-B10 GGUF metadata mismatch: "+key);
    };
    need_str("loraai.omni.stage","S1-B10");
    need_str("loraai.omni.component","code_predictor");
    need_str("loraai.omni.index_sha256",
        "03ffe9a54efe9007ceb788c2b036f5c589094918c4582add4e92bc17e1cba838");
    need_str("loraai.omni.config_sha256",
        "eab5093d47807aaf894119506b238b2b1cee70d08456e894fee9a012d88f2e0d");
    auto it=kv.find("loraai.omni.inference_ready");
    if(it==kv.end()||it->second.type!=7||it->second.flag)
        throw omni::Failure("S1-B10 staging must not claim inference ready");
    it=kv.find("loraai.omni.num_code_groups");
    if(it==kv.end()||it->second.type!=4||it->second.number!=GROUPS)
        throw omni::Failure("S1-B10 wrong code group count");
}

void cp_finite(const std::vector<float>&x){
    for(float v:x)if(!std::isfinite(v))throw omni::Failure("S1-B10 nonfinite data");
}
std::vector<float> cp_weight(omni::Reader&r,const std::string&name,
                             const std::vector<std::uint64_t>& dims){
    const auto&t=r.get_tensor(name);
    if(t.type!=omni::GGUF_BF16||t.dims!=dims)
        throw omni::Failure("S1-B10 bad predictor weight dimensions: "+name);
    auto v=to_f32(r.read_region(t.offset,static_cast<std::size_t>(t.bytes)));
    cp_finite(v);
    return v;
}
std::vector<float> cp_mul(const std::vector<float>& w,const std::vector<float>& x,
                           int ni,int no,int threads){
    if(w.size()!=static_cast<std::size_t>(ni)*no||x.size()!=static_cast<std::size_t>(ni))
        throw omni::Failure("S1-B10 invalid GGML linear shapes");
    ggml_init_params p{};
    p.mem_size=4*(w.size()+x.size()+no)+4*1024*1024;
    p.no_alloc=false;
    auto*ctx=ggml_init(p);
    if(!ctx)throw omni::Failure("S1-B10 ggml context allocation failed");
    struct Guard {ggml_context*ctx;~Guard(){ggml_free(ctx);}} guard{ctx};
    auto* tw=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,ni,no);
    auto* tx=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,ni,1);
    if(!tw||!tx)throw omni::Failure("S1-B10 ggml matrix allocation failed");
    std::memcpy(ggml_get_data(tw),w.data(),w.size()*4);
    std::memcpy(ggml_get_data(tx),x.data(),x.size()*4);
    auto*ty=ggml_mul_mat(ctx,tw,tx);
    auto*g=ggml_new_graph(ctx);
    ggml_build_forward_expand(g,ty);
    if(ggml_graph_compute_with_ctx(ctx,g,threads)!=GGML_STATUS_SUCCESS)
        throw omni::Failure("S1-B10 ggml_mul_mat computation failed");
    std::vector<float>y(no);
    std::memcpy(y.data(),ggml_get_data(ty),no*4);
    cp_finite(y);
    return y;
}
void cp_rms(std::vector<float>&x,const std::vector<float>&gain,int length){
    if(x.size()%length||gain.size()!=static_cast<std::size_t>(length))
        throw omni::Failure("S1-B10 RMSNorm size mismatch");
    for(std::size_t off=0;off<x.size();off+=length){
        double ss=0;
        for(int i=0;i<length;++i)ss+=double(x[off+i])*x[off+i];
        const double scale=1/std::sqrt(ss/length+EPS);
        for(int i=0;i<length;++i)x[off+i]=float(double(x[off+i])*gain[i]*scale);
    }
    cp_finite(x);
}
// Qwen3 STANDARD (not multimodal) RoPE: rotate_half [0..63] and [64..127].
// This is deliberately different from Talker's multimodal interleaved RoPE.
void cp_rope(std::vector<float>&x,int nheads,int position){
    if(x.size()!=std::size_t(nheads)*HEAD||position<0)
        throw omni::Failure("S1-B10 RoPE tensor geometry invalid");
    for(int head=0;head<nheads;++head){
        for(int d=0;d<HEAD/2;++d){
            const double angle=position/std::pow(THETA,double(2*d)/HEAD);
            const double c=std::cos(angle),s=std::sin(angle);
            const std::size_t i=std::size_t(head)*HEAD+d;
            const float first=x[i],second=x[i+HEAD/2];
            x[i]=float(first*c-second*s);
            x[i+HEAD/2]=float(second*c+first*s);
        }
    }
    cp_finite(x);
}
std::vector<float> cp_attention(const std::vector<float>&q,
        const std::vector<std::vector<float>>&keys,
        const std::vector<std::vector<float>>&values){
    if(q.size()!=QDIM||keys.size()!=values.size()||keys.empty()||keys.size()>3)
        throw omni::Failure("S1-B10 attention cache geometry invalid");
    std::vector<float>out(QDIM,0);
    for(int h=0;h<QHEADS;++h){
        const int kv=h/(QHEADS/KVHEADS);
        std::vector<double>scores;
        double maxscore=-std::numeric_limits<double>::infinity();
        for(std::size_t t=0;t<keys.size();++t){
            if(keys[t].size()!=KVDIM||values[t].size()!=KVDIM)
                throw omni::Failure("S1-B10 truncated KV-cache");
            double dot=0;
            for(int i=0;i<HEAD;++i)
                dot+=double(q[h*HEAD+i])*keys[t][kv*HEAD+i];
            double sc=dot/std::sqrt(double(HEAD));
            scores.push_back(sc);maxscore=std::max(maxscore,sc);
        }
        double denominator=0;
        for(auto&score:scores){score=std::exp(score-maxscore);denominator+=score;}
        for(std::size_t t=0;t<scores.size();++t){
            const double p=scores[t]/denominator;
            for(int i=0;i<HEAD;++i)
                out[h*HEAD+i]+=float(p*values[t][kv*HEAD+i]);
        }
    }
    cp_finite(out);
    return out;
}
void cp_add(std::vector<float>&a,const std::vector<float>&b){
    if(a.size()!=b.size())throw omni::Failure("S1-B10 residual size mismatch");
    for(std::size_t i=0;i<a.size();++i)a[i]+=b[i];
    cp_finite(a);
}
struct CPBlock {
    std::vector<float> norm1,norm2,qn,kn,q,k,v,o,gate,up,down;
    explicit CPBlock(omni::Reader&r,int index){
        const auto s="talker.code_predictor.model.layers."+std::to_string(index)+".";
        norm1=cp_weight(r,s+"input_layernorm.weight",{H});
        norm2=cp_weight(r,s+"post_attention_layernorm.weight",{H});
        qn=cp_weight(r,s+"self_attn.q_norm.weight",{HEAD});
        kn=cp_weight(r,s+"self_attn.k_norm.weight",{HEAD});
        q=cp_weight(r,s+"self_attn.q_proj.weight",{H,QDIM});
        k=cp_weight(r,s+"self_attn.k_proj.weight",{H,KVDIM});
        v=cp_weight(r,s+"self_attn.v_proj.weight",{H,KVDIM});
        o=cp_weight(r,s+"self_attn.o_proj.weight",{QDIM,H});
        gate=cp_weight(r,s+"mlp.gate_proj.weight",{H,MID});
        up=cp_weight(r,s+"mlp.up_proj.weight",{H,MID});
        down=cp_weight(r,s+"mlp.down_proj.weight",{MID,H});
    }
    std::array<std::vector<float>,2> forward(const std::array<std::vector<float>,2>&input,
                                              int threads){
        std::vector<std::vector<float>>cache_k,cache_v;
        std::array<std::vector<float>,2>outputs;
        for(int t=0;t<2;++t){
            std::vector<float>nx=input[t];cp_rms(nx,norm1,H);
            auto qp=cp_mul(q,nx,H,QDIM,threads);
            auto kp=cp_mul(k,nx,H,KVDIM,threads);
            auto vp=cp_mul(v,nx,H,KVDIM,threads);
            cp_rms(qp,qn,HEAD);cp_rms(kp,kn,HEAD);
            cp_rope(qp,QHEADS,t);cp_rope(kp,KVHEADS,t);
            cache_k.push_back(std::move(kp));cache_v.push_back(std::move(vp));
            auto att=cp_attention(qp,cache_k,cache_v);
            auto out=cp_mul(o,att,QDIM,H,threads);
            cp_add(out,input[t]);
            std::vector<float>norm=out;cp_rms(norm,norm2,H);
            auto gw=cp_mul(gate,norm,H,MID,threads);
            const auto uw=cp_mul(up,norm,H,MID,threads);
            for(int i=0;i<MID;++i)gw[i]=float(double(gw[i])/(1+std::exp(-double(gw[i])))*uw[i]);
            auto mlp=cp_mul(down,gw,MID,H,threads);
            cp_add(out,mlp);
            outputs[t]=std::move(out);
        }
        return outputs;
    }
};
std::vector<float> cp_load_tail(const fs::path&p,int steps){
    if(fs::is_symlink(p)||!fs::is_regular_file(p)||
       fs::file_size(p)!=std::uintmax_t(steps*H*4))
        throw omni::Failure("S1-B10 input f32 missing or wrong size: "+p.string());
    std::ifstream f(p,std::ios::binary);
    f.seekg(std::streamoff((steps-1)*H*4));
    std::vector<float>x(H);
    f.read(reinterpret_cast<char*>(x.data()),H*4);
    if(!f)throw omni::Failure("S1-B10 input read error");
    cp_finite(x);return x;
}
std::string cp_hash(const std::vector<float>&v){
    std::uint64_t h=14695981039346656037ULL;
    const auto*p=reinterpret_cast<const unsigned char*>(v.data());
    for(std::size_t i=0;i<v.size()*4;++i){h^=p[i];h*=1099511628211ULL;}
    std::ostringstream s;s<<std::hex<<std::setfill('0')<<std::setw(16)<<h;
    return s.str();
}
void cp_write_vec(const fs::path&p,const std::vector<float>&v){
    std::ofstream f(p,std::ios::binary|std::ios::trunc);
    if(!f)throw omni::Failure("S1-B10 output creation failed");
    f.write(reinterpret_cast<const char*>(v.data()),std::streamsize(v.size()*4));
    if(!f.good())throw omni::Failure("S1-B10 output write failed");
}
std::vector<std::string> cp_output_names(const CPArgs&a){
    std::vector<std::string> names;
    for(int l=0;l<LAYERS;++l)names.push_back(a.out.string()+".layer"+std::to_string(l)+".f32");
    names.push_back(a.out.string()+".norm.f32");
    names.push_back(a.out.string()+".logits.f32");
    names.push_back(a.out.string()+".json");
    return names;
}
int cp_self_test(){
    const auto got=cp_mul({.2f,.1f,-.3f,.4f,.2f,.1f},
                           {.5f,-.25f,.125f},3,2,1);
    if(std::abs(got[0]-(.2*.5+.1*-.25-.3*.125))>1e-6 ||
       std::abs(got[1]-(.4*.5+.2*-.25+.1*.125))>1e-6)
        throw omni::Failure("S1-B10 GGML transposition self-test failed");
    std::vector<float>v(QDIM,0);v[0]=.12f;v[64]=-.08f;
    const auto old=v;cp_rope(v,QHEADS,0);
    if(v!=old)throw omni::Failure("S1-B10 standard RoPE pos0 check failed");
    cp_rope(v,QHEADS,1);
    if(v==old||!std::isfinite(v[0]))throw omni::Failure("S1-B10 RoPE pos1 check failed");
    std::cout<<"S1-B10 GGML linear / standard split-half RoPE / GQA contract: PASS\n";
    return 0;
}
int cp_run(const CPArgs&a){
    if(a.self_test)return cp_self_test();
    if(!fs::is_directory(a.out.parent_path()))throw omni::Failure("S1-B10 output directory missing");
    for(const auto&name:cp_output_names(a)){
        if(fs::exists(name)||fs::is_symlink(name))throw omni::Failure("S1-B10 refuses overwrite: "+name);
        const auto path=fs::weakly_canonical(name);
        for(const auto&p:{a.model,a.hidden,a.embedding,a.baseline})
            if(path==fs::weakly_canonical(p))throw omni::Failure("S1-B10 output aliases input");
    }
    omni::Reader r(a.model);
    cp_verify_layout(r,a.baseline);
    auto first=cp_load_tail(a.hidden,a.steps),second=cp_load_tail(a.embedding,a.steps);
    const auto input_hash=cp_hash(first)+":"+cp_hash(second);
    std::array<std::vector<float>,2>inputs={std::move(first),std::move(second)};
    const auto begin=std::chrono::steady_clock::now();
    std::vector<std::vector<float>>stages;
    for(int layer=0;layer<LAYERS;++layer){
        CPBlock block(r,layer);
        inputs=block.forward(inputs,a.threads);
        stages.push_back(inputs[1]);
    }
    auto norm=inputs[1];
    cp_rms(norm,cp_weight(r,"talker.code_predictor.model.norm.weight",{H}),H);
    auto logits=cp_mul(cp_weight(r,"talker.code_predictor.lm_head.0.weight",{H,CODEVOCAB}),
                       norm,H,CODEVOCAB,a.threads);
    const auto argmax=int(std::max_element(logits.begin(),logits.end())-logits.begin());
    if(*std::max_element(norm.begin(),norm.end())-*std::min_element(norm.begin(),norm.end())<1e-8f ||
       *std::max_element(logits.begin(),logits.end())-*std::min_element(logits.begin(),logits.end())<1e-8f)
        throw omni::Failure("S1-B10 degenerate predictor output; refusing false success");
    const double elapsed=std::chrono::duration<double>(
        std::chrono::steady_clock::now()-begin).count();
    std::vector<fs::path>created;
    try{
        for(int layer=0;layer<LAYERS;++layer){
            fs::path p=a.out.string()+".layer"+std::to_string(layer)+".f32";
            created.push_back(p);cp_write_vec(p,stages[layer]);
        }
        created.emplace_back(a.out.string()+".norm.f32");cp_write_vec(created.back(),norm);
        created.emplace_back(a.out.string()+".logits.f32");cp_write_vec(created.back(),logits);
        created.emplace_back(a.out.string()+".json");
        std::ofstream f(created.back());
        if(!f)throw omni::Failure("S1-B10 report creation failed");
        f<<std::setprecision(13)
          <<"{\"schema\":\"lora.omni.s1b10.predictor.v1\",\"ok\":true,"
          <<"\"stage\":\"S1-B10-five-layer-predictor-first-residual-head\","
          <<"\"inference_ready\":false,\"upstream\":\"synthetic-B9-conditioning\","
          <<"\"source_model\":"<<omni::quote(a.model.string())
          <<",\"hidden_file\":"<<omni::quote(a.hidden.string())
          <<",\"codec_embedding_file\":"<<omni::quote(a.embedding.string())
          <<",\"steps\":"<<a.steps<<",\"codec0_id\":"<<a.codec0
          <<",\"input_hash\":"<<omni::quote(input_hash)
          <<",\"layers\":5,\"num_code_groups\":16,\"code_predictor_head\":0,"
          <<"\"residual_codec_id\":"<<argmax<<",\"vocab_size\":2048,"
          <<"\"elapsed_seconds\":"<<elapsed
          <<",\"layer_outputs\":[";
        for(int i=0;i<LAYERS;++i){if(i)f<<",";f<<omni::quote(a.out.string()+".layer"+std::to_string(i)+".f32");}
        f<<"],\"norm_file\":"<<omni::quote(a.out.string()+".norm.f32")
         <<",\"logits_file\":"<<omni::quote(a.out.string()+".logits.f32")
         <<",\"logits_max\":"<<*std::max_element(logits.begin(),logits.end())
         <<",\"limitations\":\"Test-only conditioning; ONE residual code group, not 15; no native Thinker bridge, no Code2Wav or audio.\"}\n";
        if(!f.good())throw omni::Failure("S1-B10 report write failure");
    }catch(...){
        for(const auto&p:created){std::error_code ec;fs::remove(p,ec);}
        throw;
    }
    std::cout<<"S1-B10 predictor layers=5 first residual codec="<<argmax
             <<" (BF16 GGUF / GGML+CPU, synthetic upstream, NOT speech); elapsed="
             <<elapsed<<"s\nReport: "<<a.out.string()<<".json\n";
    return 0;
}
} // anonymous namespace