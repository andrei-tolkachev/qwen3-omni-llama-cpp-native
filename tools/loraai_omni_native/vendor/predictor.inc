// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andrei Tolkachev
// Derived from the user-provided, BF16-verified LoraAI kernels. See provenance.json.
// LoraAI Qwen3-Omni S1-B14L: original BF16, second REAL Q4 Thinker->Talker codec frame.
// BF16 staging GGUF + GGML CPU matmul; deterministic greedy RVQ generation.
// Reuses audited S1-B10 tensor reader/math without modifying that source.
// Genuine B14K codec feedback terminal state; second 16-group greedy frame. NOT speech or full-model Transformers parity.
#include "predictor_base.inc"

namespace {
constexpr int AR_GROUPS=15; // Talker group 0 + Code Predictor groups 1..15
constexpr int AR_POSITIONS=16; // two prefills + fourteen residual embedding positions

struct ARArgs {
    fs::path model="omni-experimental/code-predictor-staging-bf16.gguf";
    fs::path baseline,hidden,embedding,output;
    int steps=1,threads=2,codec0=-1,expected_codec1=-1;
    bool self_test=false;
};
ARArgs ar_args(int argc,char**argv) {
    ARArgs a;
    for(int i=1;i<argc;++i){
        const std::string k=argv[i];
        if(k=="--self-test") a.self_test=true;
        else if(k=="--model" && i+1<argc) a.model=argv[++i];
        else if(k=="--baseline" && i+1<argc) a.baseline=argv[++i];
        else if(k=="--talker-hidden" && i+1<argc) a.hidden=argv[++i];
        else if(k=="--codec0-embedding" && i+1<argc) a.embedding=argv[++i];
        else if(k=="--codec0-id" && i+1<argc) a.codec0=std::stoi(argv[++i]);
        else if(k=="--expected-codec1" && i+1<argc) a.expected_codec1=std::stoi(argv[++i]);
        else if(k=="--steps" && i+1<argc) a.steps=std::stoi(argv[++i]);
        else if(k=="--threads" && i+1<argc) a.threads=std::stoi(argv[++i]);
        else if(k=="--output-prefix" && i+1<argc) a.output=argv[++i];
        else if(k=="--help"){
            std::cout<<"omni_code_predictor_ar_ggml --model GGUF --baseline TSV --talker-hidden F32 "
                     "--codec0-embedding F32 --codec0-id N --expected-codec1 N "
                     "--output-prefix PATH [--steps 1] [--threads N]\n"
                     "S1-B14L: 15 residual codec groups from GENUINE B14K terminal hidden state. ONE frame. NOT speech.\n";
            std::exit(0);
        } else throw omni::Failure("S1-B14L unexpected/partial argument: "+k);
    }
    if(a.steps!=1 || a.threads<1 || a.threads>16 ||
       (!a.self_test && (a.baseline.empty() || a.hidden.empty() || a.embedding.empty() ||
                         a.output.empty() || a.codec0<0 || a.codec0>=2048 ||
                         a.expected_codec1<-1 || a.expected_codec1>=CODEVOCAB)))
        throw omni::Failure("S1-B14L missing/invalid arguments");
    return a;
}
// Same numerical kernel as S1-B10, but retains up to all 16 causal positions.
std::vector<float> ar_attend(const std::vector<float>&q,
        const std::vector<std::vector<float>>&keys,
        const std::vector<std::vector<float>>&values) {
    if(q.size()!=QDIM || keys.empty() || keys.size()>AR_POSITIONS || keys.size()!=values.size())
        throw omni::Failure("S1-B14L incorrect KV chronology");
    std::vector<float> out(QDIM,0);
    for(int head=0;head<QHEADS;++head){
        const int kv=head/(QHEADS/KVHEADS);
        std::vector<double> scores;
        scores.reserve(keys.size());
        double greatest=-std::numeric_limits<double>::infinity();
        for(std::size_t t=0;t<keys.size();++t){
            if(keys[t].size()!=KVDIM || values[t].size()!=KVDIM)
                throw omni::Failure("S1-B14L malformed cached K/V");
            double dot=0;
            for(int i=0;i<HEAD;++i)dot+=double(q[head*HEAD+i])*keys[t][kv*HEAD+i];
            const double sc=dot/std::sqrt(double(HEAD));
            scores.push_back(sc);greatest=std::max(greatest,sc);
        }
        double total=0;
        for(auto&sc:scores){sc=std::exp(sc-greatest);total+=sc;}
        if(!(total>0) || !std::isfinite(total))throw omni::Failure("S1-B14L invalid attention softmax");
        for(std::size_t t=0;t<scores.size();++t){
            const double p=scores[t]/total;
            for(int i=0;i<HEAD;++i)out[head*HEAD+i]+=float(p*values[t][kv*HEAD+i]);
        }
    }
    cp_finite(out);return out;
}
struct ARBlock {
    CPBlock weights;
    std::vector<std::vector<float>> key_cache,value_cache;
    explicit ARBlock(omni::Reader&r,int layer):weights(r,layer) {}
    std::vector<float> forward(const std::vector<float>&input,int position,int threads){
        if(input.size()!=H || key_cache.size()!=std::size_t(position) ||
           value_cache.size()!=std::size_t(position) || position>=AR_POSITIONS)
            throw omni::Failure("S1-B14L layer-specific KV-cache out of sequence");
        auto normalized=input;cp_rms(normalized,weights.norm1,H);
        auto q=cp_mul(weights.q,normalized,H,QDIM,threads);
        auto k=cp_mul(weights.k,normalized,H,KVDIM,threads);
        auto v=cp_mul(weights.v,normalized,H,KVDIM,threads);
        cp_rms(q,weights.qn,HEAD);cp_rms(k,weights.kn,HEAD);
        cp_rope(q,QHEADS,position);cp_rope(k,KVHEADS,position);
        key_cache.push_back(std::move(k));value_cache.push_back(std::move(v));
        auto ctx=ar_attend(q,key_cache,value_cache);
        auto output=cp_mul(weights.o,ctx,QDIM,H,threads);
        cp_add(output,input);
        auto post=output;cp_rms(post,weights.norm2,H);
        auto gate=cp_mul(weights.gate,post,H,MID,threads);
        auto up=cp_mul(weights.up,post,H,MID,threads);
        for(int i=0;i<MID;++i)
            gate[i]=float(double(gate[i])/(1.0+std::exp(-double(gate[i])))*up[i]);
        auto down=cp_mul(weights.down,gate,MID,H,threads);
        cp_add(output,down);
        return output;
    }
};
std::vector<float> ar_lookup(omni::Reader&r,int table,int codec){
    if(table<0 || table>=AR_GROUPS || codec<0 || codec>=CODEVOCAB)
        throw omni::Failure("S1-B14L codec embedding index out of bounds");
    const std::string name="talker.code_predictor.model.codec_embedding."+std::to_string(table)+".weight";
    const auto&t=r.get_tensor(name);
    if(t.type!=omni::GGUF_BF16 || t.dims!=std::vector<std::uint64_t>{H,CODEVOCAB})
        throw omni::Failure("S1-B14L incorrect group codec embedding dimensions: "+name);
    const std::uint64_t off=t.offset+std::uint64_t(codec)*H*2;
    if(off<t.offset || off+H*2>t.offset+t.bytes)
        throw omni::Failure("S1-B14L codec embedding outside tensor");
    auto v=to_f32(r.read_region(off,H*2));cp_finite(v);return v;
}

}
