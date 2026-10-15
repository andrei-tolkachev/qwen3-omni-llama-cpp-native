// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andrei Tolkachev
// Derived from the user-provided, BF16-verified LoraAI kernels. See provenance.json.
// LoraAI Qwen3-Omni S1-B13D1. Variable-frame DAC input stem, controlled codec codes ONLY.
// Full B13C 2-stage validated N*4 x 1024 input -> causal Conv1d(k=7) 1536 -> SnakeBeta 1536.
// Not four decoder cascades, PCM/WAV, speech inference, CUDA or a Thinker->Talker bridge.
#include "reader.inc"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
namespace fs = std::filesystem;
namespace {
using Vec = std::vector<float>;
constexpr int INPUT_CH=1024, OUTPUT_CH=1536, KERNEL=7, MAX_CODEC_FRAMES=512;
constexpr int MAX_DAC_FRAMES=4*MAX_CODEC_FRAMES;
const char* MODEL_DEFAULT="omni-experimental/code2wav-staging-bf16.gguf";
struct Args {
 fs::path model=MODEL_DEFAULT,input,out;
 int threads=2,expected_frames=0;
 bool self_test=false;
};
void check(bool ok,const std::string& reason){if(!ok)throw omni::Failure(reason);}
void finite(const Vec& v){for(float f:v)check(std::isfinite(f),"nonfinite native DAC tensor");}
Args parse(int argc,char**argv){Args a;for(int i=1;i<argc;++i){std::string s=argv[i];
 if(s=="--self-test")a.self_test=true;
 else if(s=="--model"&&i+1<argc)a.model=argv[++i];
 else if(s=="--input-f32"&&i+1<argc)a.input=argv[++i];
 else if(s=="--out"&&i+1<argc)a.out=argv[++i];
 else if(s=="--threads"&&i+1<argc)a.threads=std::stoi(argv[++i]);
 else if(s=="--expected-frames"&&i+1<argc)a.expected_frames=std::stoi(argv[++i]);
 else throw omni::Failure("unknown/incomplete DAC option: "+s);
 }
 check(a.threads>=1&&a.threads<=32,"threads must be 1..32");
 if(!a.self_test){
  check(!a.input.empty()&&!a.out.empty(),"--input-f32 and --out required");
  check(a.expected_frames>=4 && a.expected_frames<=MAX_DAC_FRAMES && a.expected_frames%4==0,
        "DAC frames must be 4*N with 1<=N<=512");
 }
 return a;
}
Vec get(omni::Reader& r,const std::string& name,const std::vector<std::uint64_t>& dims){
 const auto& t=r.get_tensor(name);
 check(t.type==omni::GGUF_BF16 && t.dims==dims,"wrong BF16 GGUF shape/type: "+name);
 auto raw=r.read_region(t.offset,static_cast<size_t>(t.bytes));
 check(raw.size()==t.bytes && raw.size()%2==0,"truncated BF16 GGUF tensor: "+name);
 Vec result(raw.size()/2);
 for(size_t i=0;i<result.size();++i){
  std::uint16_t v=std::uint16_t(raw[2*i])|(std::uint16_t(raw[2*i+1])<<8);
  std::uint32_t bits=std::uint32_t(v)<<16;std::memcpy(&result[i],&bits,sizeof(float));
 }
 finite(result);return result;
}
void provenance(omni::Reader&r){
 check(r.tensors().size()==230 && r.payload_bytes()==432033154ULL,"wrong pinned Code2Wav tensor inventory");
 check(omni::require_string(r,"loraai.omni.stage")=="S1-B12A" &&
       omni::require_string(r,"loraai.omni.component")=="code2wav" &&
       omni::require_string(r,"loraai.omni.config_sha256")=="eab5093d47807aaf894119506b238b2b1cee70d08456e894fee9a012d88f2e0d",
       "wrong staging GGUF provenance");
 auto it=r.metadata().find("loraai.omni.inference_ready");
 check(it!=r.metadata().end() && it->second.type==7 && !it->second.flag,
       "inference_ready must remain false for staging GGUF");
}
Vec load_input(const fs::path&p,int frames){
 check(!fs::is_symlink(p),"aliased DAC input prohibited");
 std::ifstream f(p,std::ios::binary|std::ios::ate);check(bool(f),"cannot open accepted B13C output");
 check(f.tellg()==std::streamoff(size_t(frames)*INPUT_CH*sizeof(float)),"DAC input must equal expected (4*N)x1024");
 f.seekg(0);Vec x(size_t(frames)*INPUT_CH);
 f.read(reinterpret_cast<char*>(x.data()),static_cast<std::streamsize>(x.size()*sizeof(float)));
 check(bool(f),"truncated DAC input");finite(x);return x;
}
// HF Conv1d [out,in,k] in C-contiguous order: im2col gives [t][in*k] in same order.
// Strict left-zero-padding KERNEL-1, no future frames or symmetric padding.
Vec im2col(const Vec& x,int frames,int inch,int kernel){
 check(frames>0&&inch>0&&kernel>0&&x.size()==size_t(frames)*inch,"im2col geometry invalid");
 Vec col(size_t(frames)*inch*kernel,0.f);
 for(int t=0;t<frames;++t)for(int c=0;c<inch;++c)for(int k=0;k<kernel;++k){
  int input_t=t-(kernel-1)+k;
  if(input_t>=0)col[(size_t(t)*inch+c)*kernel+k]=x[size_t(input_t)*inch+c];
 }
 return col;
}
Vec gemm(const Vec& w,const Vec& cols,int in,int out,int frames,int threads){
 check(w.size()==size_t(in)*out&&cols.size()==size_t(in)*frames,"DAC GGML matrix geometry invalid");
 ggml_init_params p{};
 p.mem_size=size_t(16)*1024*1024+(w.size()+cols.size()+size_t(frames)*out)*sizeof(float);
 p.no_alloc=false;
 ggml_context* ctx=ggml_init(p);check(ctx!=nullptr,"DAC ggml_init failed");
 struct Cleanup{ggml_context*ctx;~Cleanup(){ggml_free(ctx);}}cleanup{ctx};
 auto* W=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,in,out);
 auto* X=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,in,frames);
 check(W&&X,"DAC ggml tensor allocation failed");
 std::memcpy(ggml_get_data(W),w.data(),w.size()*sizeof(float));
 std::memcpy(ggml_get_data(X),cols.data(),cols.size()*sizeof(float));
 auto* Y=ggml_mul_mat(ctx,W,X);check(Y!=nullptr,"DAC ggml_mul_mat failed");
 auto* g=ggml_new_graph(ctx);check(g!=nullptr,"DAC ggml graph creation failed");
 ggml_build_forward_expand(g,Y);
 check(ggml_graph_compute_with_ctx(ctx,g,threads)==GGML_STATUS_SUCCESS,"DAC ggml matrix execution failed");
 Vec y(size_t(frames)*out);
 std::memcpy(y.data(),ggml_get_data(Y),y.size()*sizeof(float));finite(y);return y;
}
Vec causal_conv(const Vec& x,const Vec& w,const Vec& b,int frames,int inch,int outch,int kernel,int threads){
 check(x.size()==size_t(frames)*inch && w.size()==size_t(outch)*inch*kernel && b.size()==size_t(outch),
       "DAC Conv1d input/weight/bias geometry invalid");
 auto cols=im2col(x,frames,inch,kernel);
 auto result=gemm(w,cols,inch*kernel,outch,frames,threads);
 for(int t=0;t<frames;++t)for(int o=0;o<outch;++o)result[size_t(t)*outch+o]+=b[o];
 finite(result);return result;
}
// Qwen3-Omni SnakeBeta: x + sin(x*exp(alpha))^2/(exp(beta)+1e-9).
Vec snake(const Vec& x,const Vec& alpha,const Vec& beta,int frames,int channels){
 check(x.size()==size_t(frames)*channels && alpha.size()==size_t(channels) && beta.size()==size_t(channels),
       "DAC SnakeBeta geometry invalid");
 Vec result(x.size());
 for(int t=0;t<frames;++t)for(int c=0;c<channels;++c){
  size_t i=size_t(t)*channels+c;
  double v=x[i], wave=std::sin(v*std::exp(double(alpha[c])));
  result[i]=float(v+wave*wave/(std::exp(double(beta[c]))+1e-9));
 }
 finite(result);return result;
}

}
