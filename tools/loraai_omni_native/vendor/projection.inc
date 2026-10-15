// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andrei Tolkachev
// Derived from the user-provided, BF16-verified LoraAI kernels. See provenance.json.
// S1-B14A: real BF16 input projections for Qwen3-Omni Talker (text and hidden).
// Controlled synthetic 2048-dimensional inputs ONLY, NOT Thinker-derived hidden states.
// Native GGML BF16->F32 GEMM, SiLU, second GEMM; independent HF BF16 comparison required.
#include "reader.inc"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {
using V=std::vector<float>;
constexpr int INPUT=2048, INTERMEDIATE=2048, OUTPUT=1024, MAX_STEPS=16;
struct Args { fs::path model="omni-experimental/talker-experts-packed-bf16-staging.gguf";
 fs::path baseline,input,prefix;std::string mode;int steps=0,threads=2;bool self_test=false;};
void insist(bool good,const std::string&err){if(!good)throw omni::Failure("S1-B14A "+err);}
void finite(const V&v){for(float x:v)insist(std::isfinite(x),"nonfinite values");}
Args parse(int argc,char**argv){Args a;
 for(int i=1;i<argc;++i){std::string s=argv[i];
  if(s=="--self-test")a.self_test=true;
  else if(s=="--model"&&i+1<argc)a.model=argv[++i];
  else if(s=="--baseline"&&i+1<argc)a.baseline=argv[++i];
  else if(s=="--mode"&&i+1<argc)a.mode=argv[++i];
  else if(s=="--input-f32"&&i+1<argc)a.input=argv[++i];
  else if(s=="--output-prefix"&&i+1<argc)a.prefix=argv[++i];
  else if(s=="--steps"&&i+1<argc)a.steps=std::stoi(argv[++i]);
  else if(s=="--threads"&&i+1<argc)a.threads=std::stoi(argv[++i]);
  else if(s=="--help"){std::cout<<"S1-B14A --self-test | --model GGUF --baseline TSV --mode text|hidden --input-f32 IN --steps 1..16 --output-prefix PREFIX [--threads 2] (CONTROLLED INPUT, NOT SPEECH)\n";std::exit(0);}
  else throw omni::Failure("S1-B14A unknown/incomplete option: "+s);
 }
 insist(a.threads>=1&&a.threads<=16,"invalid thread count");
 if(a.self_test)insist(a.mode.empty()&&a.input.empty()&&a.prefix.empty()&&a.steps==0,"self-test must not write output");
 else insist((a.mode=="text"||a.mode=="hidden")&&a.steps>=1&&a.steps<=MAX_STEPS&&!a.baseline.empty()&&!a.input.empty()&&!a.prefix.empty(),"missing/invalid mode, steps or required paths");
 return a;
}
V read_input(const fs::path&p,int steps){
 insist(fs::is_regular_file(p)&&!fs::is_symlink(p),"input missing or symlinked");
 const auto n=size_t(steps)*INPUT;
 insist(fs::file_size(p)==n*sizeof(float),"input must be exact steps x 2048 F32");
 V v(n);std::ifstream f(p,std::ios::binary);f.read(reinterpret_cast<char*>(v.data()),std::streamsize(n*4));
 insist(bool(f),"input truncated");finite(v);return v;
}
V read_bf16(omni::Reader&rd,const std::string&name,const std::vector<std::uint64_t>&dims){
 const auto&t=rd.get_tensor(name);
 insist(t.type==omni::GGUF_BF16&&t.dims==dims,"weight shape/BF16 mismatch: "+name);
 auto raw=rd.read_region(t.offset,size_t(t.bytes));V v(raw.size()/2);
 for(size_t i=0;i<v.size();++i){auto b=std::uint16_t(raw[2*i])|(std::uint16_t(raw[2*i+1])<<8);
  std::uint32_t bits=std::uint32_t(b)<<16;std::memcpy(&v[i],&bits,4);}
 finite(v);return v;
}
V gemm(const V&w,const V&x,int ni,int no,int steps,int threads){
 insist(int(w.size())==ni*no&&int(x.size())==ni*steps&&steps>0,"GGML matmul shape mismatch");
 ggml_init_params p{};p.mem_size=size_t(16)*1024*1024+(w.size()+x.size()+size_t(no)*steps)*4;p.no_alloc=false;
 auto*ctx=ggml_init(p);insist(ctx!=nullptr,"ggml_init failed");
 struct Cleanup{ggml_context*p;~Cleanup(){ggml_free(p);}} cleanup{ctx};
 auto*W=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,ni,no);
 auto*X=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,ni,steps);
 insist(W&&X,"GGML tensor creation failed");
 std::memcpy(ggml_get_data(W),w.data(),w.size()*4);
 std::memcpy(ggml_get_data(X),x.data(),x.size()*4);
 auto*Y=ggml_mul_mat(ctx,W,X);insist(Y!=nullptr,"GGML matmul failed");
 auto*g=ggml_new_graph(ctx);insist(g!=nullptr,"GGML graph failed");ggml_build_forward_expand(g,Y);
 insist(ggml_graph_compute_with_ctx(ctx,g,threads)==GGML_STATUS_SUCCESS,"GGML graph compute failed");
 V result(size_t(no)*steps);std::memcpy(result.data(),ggml_get_data(Y),result.size()*4);
 finite(result);return result;
}
float silu(float x){if(x>=0)return x/(1.0f+std::exp(-x));float e=std::exp(x);return x*e/(1.0f+e);}
struct Stages{V fc1,activated,out;};
Stages compute(const V&x,const V&w1,const V&b1,const V&w2,const V&b2,int ni,int mid,int no,int steps,int threads){
 insist(b1.size()==size_t(mid)&&b2.size()==size_t(no),"linear bias shape mismatch");
 Stages s;s.fc1=gemm(w1,x,ni,mid,steps,threads);
 for(int t=0;t<steps;++t)for(int j=0;j<mid;++j)s.fc1[size_t(t)*mid+j]+=b1[j];
 finite(s.fc1);s.activated=s.fc1;
 for(float&v:s.activated)v=silu(v);
 finite(s.activated);
 s.out=gemm(w2,s.activated,mid,no,steps,threads);
 for(int t=0;t<steps;++t)for(int j=0;j<no;++j)s.out[size_t(t)*no+j]+=b2[j];
 finite(s.out);return s;
}

}
