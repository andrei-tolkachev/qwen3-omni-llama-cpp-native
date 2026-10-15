// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andrei Tolkachev
// Derived from the user-provided, BF16-verified LoraAI kernels. See provenance.json.
// Generated from pinned S1-B3 numerical microtest: main() removed.
// Temporary staging reuse, replaced with reusable C++ library in subsequent S1-B5.
// S1-B3: CPU-only GGML computation of ONE native Qwen3-Omni Talker routed expert.
// BF16 packed GGUF expert slices are validated using S1-B2's pinned metadata reader.
// This is an isolated numerical microtest, NOT inference-ready Talker or llama-server.
// The independent S1-B2 reader is temporarily embedded until a shared API exists.
#include "reader.inc"

#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstring>
#include <memory>
#include <chrono>

namespace {

struct Params {
    std::filesystem::path model = "omni-experimental/talker-experts-packed-bf16-staging.gguf";
    std::filesystem::path baseline;
    std::filesystem::path output_prefix;
    int layer=0;
    int expert=7;
    int threads=2;
    bool self_test=false;
};

Params args(int argc, char **argv) {
    Params a;
    for (int i=1;i<argc;++i) {
        std::string x=argv[i];
        if (x=="--help") {
            std::cout << "omni_expert_ggml [--self-test] [--model PACKED_GGUF --baseline layout.tsv --layer 0 --expert 7 --output-prefix /tmp/name] [--threads 2]\n";
            std::exit(0);
        } else if (x=="--self-test") a.self_test=true;
        else if (x=="--model" && i+1<argc) a.model=argv[++i];
        else if (x=="--baseline" && i+1<argc) a.baseline=argv[++i];
        else if (x=="--output-prefix" && i+1<argc) a.output_prefix=argv[++i];
        else if (x=="--layer" && i+1<argc) a.layer=std::stoi(argv[++i]);
        else if (x=="--expert" && i+1<argc) a.expert=std::stoi(argv[++i]);
        else if (x=="--threads" && i+1<argc) a.threads=std::stoi(argv[++i]);
        else throw omni::Failure("unknown or incomplete argument: " + x);
    }
    if (a.threads<1 || a.threads>16 || a.layer<0 || a.layer>=20 || a.expert<0 || a.expert>=128)
        throw omni::Failure("invalid layer/expert/threads");
    if (!a.self_test && (a.baseline.empty() || a.output_prefix.empty()))
        throw omni::Failure("--baseline and --output-prefix required for real weights");
    return a;
}

std::vector<float> to_f32(const std::vector<std::uint8_t>& data) {
    if (data.size()%2) throw omni::Failure("unaligned BF16 data");
    std::vector<float> result(data.size()/2);
    for (std::size_t i=0;i<result.size();++i) {
        const std::uint16_t bf16=static_cast<std::uint16_t>(data[2*i]) | (static_cast<std::uint16_t>(data[2*i+1])<<8);
        const std::uint32_t bits=std::uint32_t(bf16)<<16;
        std::memcpy(&result[i],&bits,sizeof(float));
        if (!std::isfinite(result[i])) throw omni::Failure("nonfinite BF16 weight");
    }
    return result;
}

struct Weights {
    std::int64_t hidden=0, intermediate=0;
    std::vector<float> gate, up, down;
};

Weights load_one(omni::Reader & r,int layer,int expert) {
    Weights result;
    for (const auto& proj:{"gate_proj","up_proj","down_proj"}) {
        const auto name="talker.model.layers."+std::to_string(layer)+".mlp.experts_packed."+proj+".weight";
        const auto& t=r.get_tensor(name);
        if (t.dims.size()!=3 || t.dims[2]!=128) throw omni::Failure("wrong expert geometry");
        const auto bytes=omni::checked_mul(omni::checked_mul(t.dims[0],t.dims[1]),2);
        const auto off=omni::checked_add(t.offset,omni::checked_mul(static_cast<std::uint64_t>(expert),bytes));
        const auto raw=r.read_region(off,static_cast<std::size_t>(bytes));
        if (std::strcmp(proj,"gate_proj")==0) {
            result.hidden=static_cast<std::int64_t>(t.dims[0]);
            result.intermediate=static_cast<std::int64_t>(t.dims[1]);
            result.gate=to_f32(raw);
        } else if (std::strcmp(proj,"up_proj")==0) {
            if (t.dims[0]!=std::uint64_t(result.hidden) || t.dims[1]!=std::uint64_t(result.intermediate))
                throw omni::Failure("up_proj geometry differs from gate_proj");
            result.up=to_f32(raw);
        } else {
            if (t.dims[0]!=std::uint64_t(result.intermediate) || t.dims[1]!=std::uint64_t(result.hidden))
                throw omni::Failure("down_proj geometry incompatible with gate/up");
            result.down=to_f32(raw);
        }
    }
    if (result.hidden!=1024 || result.intermediate!=384)
        throw omni::Failure("unexpected pinned Talker expert hidden/intermediate dimensions");
    return result;
}

std::vector<float> input(std::int64_t n) {
    std::vector<float> x(static_cast<std::size_t>(n));
    for (std::int64_t i=0;i<n;++i)
        x[static_cast<std::size_t>(i)] = static_cast<float>(std::sin((i+1)*0.023)*0.03125 + std::cos((i+3)*0.007)*0.015625);
    return x;
}

std::vector<float> compute(const Weights & w,const std::vector<float> & x,int threads) {
    if (w.hidden<1 || w.intermediate<1 || x.size()!=std::size_t(w.hidden) ||
        w.gate.size()!=std::size_t(w.hidden*w.intermediate) ||
        w.up.size()!=std::size_t(w.hidden*w.intermediate) ||
        w.down.size()!=std::size_t(w.hidden*w.intermediate))
        throw omni::Failure("invalid matrix shapes");
    ggml_init_params p{};
    p.mem_size=32*1024*1024;
    p.mem_buffer=nullptr;
    p.no_alloc=false;
    ggml_context * ctx=ggml_init(p);
    if (!ctx) throw omni::Failure("ggml_init failed");
    struct Guard { ggml_context *p; ~Guard(){ggml_free(p);} } guard{ctx};
    ggml_tensor * wg=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,w.hidden,w.intermediate);
    ggml_tensor * wu=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,w.hidden,w.intermediate);
    ggml_tensor * wd=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,w.intermediate,w.hidden);
    ggml_tensor * tx=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,w.hidden,1);
    if (!wg || !wu || !wd || !tx) throw omni::Failure("GGML tensor allocation failed");
    std::memcpy(ggml_get_data(wg),w.gate.data(),w.gate.size()*sizeof(float));
    std::memcpy(ggml_get_data(wu),w.up.data(),w.up.size()*sizeof(float));
    std::memcpy(ggml_get_data(wd),w.down.data(),w.down.size()*sizeof(float));
    std::memcpy(ggml_get_data(tx),x.data(),x.size()*sizeof(float));
    // Native SwiGLU expert formula (router and shared experts are OUT OF SCOPE).
    ggml_tensor * g=ggml_mul_mat(ctx,wg,tx);
    ggml_tensor * u=ggml_mul_mat(ctx,wu,tx);
    ggml_tensor * mid=ggml_mul(ctx,ggml_silu(ctx,g),u);
    ggml_tensor * out=ggml_mul_mat(ctx,wd,mid);
    ggml_cgraph * graph=ggml_new_graph(ctx);
    ggml_build_forward_expand(graph,out);
    const auto rc=ggml_graph_compute_with_ctx(ctx,graph,threads);
    if (rc!=GGML_STATUS_SUCCESS) throw omni::Failure(std::string("GGML computation failed: ")+ggml_status_to_string(rc));
    std::vector<float> y(static_cast<std::size_t>(w.hidden));
    std::memcpy(y.data(),ggml_get_data(out),y.size()*sizeof(float));
    for(float v:y) if (!std::isfinite(v)) throw omni::Failure("GGML output nonfinite");
    return y;
}

std::vector<float> reference(const Weights & w,const std::vector<float> & x) {
    std::vector<float> mid(static_cast<std::size_t>(w.intermediate));
    for (int j=0;j<w.intermediate;++j) {
        float gate=0, up=0;
        for (int i=0;i<w.hidden;++i) {
            gate+=w.gate[std::size_t(j*w.hidden+i)]*x[std::size_t(i)];
            up+=w.up[std::size_t(j*w.hidden+i)]*x[std::size_t(i)];
        }
        mid[std::size_t(j)]=gate/(1+std::exp(-gate))*up;
    }
    std::vector<float> out(static_cast<std::size_t>(w.hidden));
    for (int j=0;j<w.hidden;++j) {
        float v=0;
        for (int i=0;i<w.intermediate;++i) v+=w.down[std::size_t(j*w.intermediate+i)]*mid[std::size_t(i)];
        out[std::size_t(j)]=v;
    }
    return out;
}

int self_test(int threads) {
    Weights w;
    w.hidden=1024;w.intermediate=384;
    const auto sz=std::size_t(w.hidden*w.intermediate);
    w.gate.resize(sz);w.up.resize(sz);w.down.resize(sz);
    for (std::size_t i=0;i<sz;++i) {
        w.gate[i]=static_cast<float>((static_cast<int>(i%19)-9)*0.01);
        w.up[i]=static_cast<float>((static_cast<int>(i%23)-11)*0.01);
        w.down[i]=static_cast<float>((static_cast<int>(i%13)-6)*0.01);
    }
    auto x=input(w.hidden);
    const auto a=compute(w,x,threads),b=reference(w,x);
    float delta=0,max_val=0;
    for (std::size_t i=0;i<a.size();++i) {
        delta=std::max(delta,std::abs(a[i]-b[i]));
        max_val=std::max(max_val,std::abs(b[i]));
    }
    const bool ok=max_val>1e-8f && delta<1e-4f;
    std::cout<<"S1-B3 GGML self-test: "<<(ok?"OK":"FAILED")<<" max_abs_diff="<<delta<<" max_ref="<<max_val<<"\n";
    return ok?0:3;
}

void save_binary(const std::filesystem::path& out, const std::vector<float>& x) {
    std::ofstream f(out,std::ios::binary | std::ios::out | std::ios::trunc);
    if (!f) throw omni::Failure("cannot write output: "+out.string());
    f.write(reinterpret_cast<const char*>(x.data()),static_cast<std::streamsize>(x.size()*sizeof(float)));
    if (!f.good()) throw omni::Failure("write failed: "+out.string());
}

int run(int argc,char**argv) {
    const auto a=args(argc,argv);
    if (a.self_test) return self_test(a.threads);
    const auto f32=a.output_prefix.string()+".f32";
    const auto json=a.output_prefix.string()+".json";
    if (std::filesystem::exists(f32) || std::filesystem::exists(json))
        throw omni::Failure("output files exist; refuse overwrite");
    if (std::filesystem::absolute(f32)==std::filesystem::absolute(a.model) ||
        std::filesystem::absolute(json)==std::filesystem::absolute(a.model) ||
        std::filesystem::absolute(f32)==std::filesystem::absolute(a.baseline) ||
        std::filesystem::absolute(json)==std::filesystem::absolute(a.baseline))
        throw omni::Failure("refuse to overwrite source files");
    auto b=omni::load_baseline(a.baseline);
    omni::Reader reader(a.model);
    omni::validate(reader,b);
    auto w=load_one(reader,a.layer,a.expert);
    const auto x=input(w.hidden);
    const auto t0=std::chrono::steady_clock::now();
    auto result=compute(w,x,a.threads);
    const auto duration=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    double sum2=0, max_abs=0;
    for(float z:result) { sum2+=double(z)*z; max_abs=std::max(max_abs,double(std::abs(z))); }
    // Rejected low-signal outputs must not be accidentally marked as successful numerical evidence.
    if (max_abs < 1e-20) throw omni::Failure("near-zero model result; investigate BF16 weights before comparison");
    std::ostringstream o;
    o << "{\n\"schema\":\"lora.omni.s1b3.ggml_expert.v1\",\n"
      << "\"ok\":true,\n\"stage\":\"S1-B3-expert-only\",\n\"inference_ready\":false,\n"
      << "\"model\":"<<omni::quote(a.model.string())<<",\n"
      << "\"layer\":"<<a.layer<<",\n\"expert\":"<<a.expert<<",\n"
      << "\"hidden\":"<<w.hidden<<",\n\"intermediate\":"<<w.intermediate<<",\n"
      << "\"dtype\":\"BF16_stored_to_F32_CPU_execution\",\n"
      << "\"threads\":"<<a.threads<<",\n\"compute_seconds\":"<<duration<<",\n"
      << "\"max_abs\":"<<max_abs<<",\n\"rms\":"<<std::sqrt(sum2/result.size())<<",\n"
      << "\"f32_file\":"<<omni::quote(f32)<<"\n}\n";
    save_binary(f32,result);
    try {
        std::ofstream f(json);
        if(!f) throw omni::Failure("cannot write JSON");
        f<<o.str();
        if(!f.good()) throw omni::Failure("cannot flush JSON");
    } catch (...) {
        std::filesystem::remove(f32);
        throw;
    }
    std::cout<<"S1-B3 GGML one expert computed (NOT Talker inference). max_abs="<<max_abs<<" seconds="<<duration<<"\n"
             <<"Outputs: "<<f32<<" and "<<json<<"\n";
    return 0;
}
} // namespace

