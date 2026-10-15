// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andrei Tolkachev
// Derived from the user-provided, BF16-verified LoraAI kernels. See provenance.json.
// S1-B13A (v0.5.5): Native variable-frame 16-quantizer Code2Wav embedding ingress.
// Strict Q3OC/LE binary -> BF16 GGUF per-quantizer row -> GGML graph mean.
// These are EXTERNAL codec IDs, not proof of Thinker/Talker speech inference.
// CPU reference math retained for numerical validation.
#include "reader.inc"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
namespace {
namespace fs = std::filesystem;
constexpr int Q=16,V=2048,H=1024,MAX_FRAMES=512;
constexpr const char* NAME="code2wav.code_embedding.weight";
constexpr const char* DEFAULT_MODEL="omni-experimental/code2wav-staging-bf16.gguf";
using CodeFrame=std::array<std::uint16_t,Q>;
using Frames=std::vector<CodeFrame>;
using Vec=std::vector<float>;
void guard(bool ok,const std::string& reason){if(!ok)throw omni::Failure(reason);}
struct Args {fs::path model=DEFAULT_MODEL,frames,out;bool selftest=false;int threads=2;};
Args parse(int argc,char**argv){Args a;for(int i=1;i<argc;++i){std::string arg=argv[i];
 if(arg=="--self-test")a.selftest=true;
 else if(arg=="--model"&&i+1<argc)a.model=argv[++i];
 else if(arg=="--codec-frames"&&i+1<argc)a.frames=argv[++i];
 else if(arg=="--output-prefix"&&i+1<argc)a.out=argv[++i];
 else if(arg=="--threads"&&i+1<argc)a.threads=std::stoi(argv[++i]);
 else throw omni::Failure("Unknown/incomplete S1-B13A argument: "+arg);
 }guard(a.threads>=1&&a.threads<=16,"Threads must be 1..16");
 if(!a.selftest)guard(!a.frames.empty()&&!a.out.empty(),"--codec-frames and --output-prefix required");
 else guard(a.frames.empty()&&a.out.empty(),"Self-test disallows inputs/outputs");
 return a;}
std::uint32_t load_le_u32(const unsigned char* p){return std::uint32_t(p[0])|(std::uint32_t(p[1])<<8)|(std::uint32_t(p[2])<<16)|(std::uint32_t(p[3])<<24);}
std::uint16_t load_le_u16(const unsigned char* p){return std::uint16_t(p[0])|std::uint16_t(p[1])<<8;}
Frames read_frames(const fs::path& path){
 guard(fs::exists(path)&&fs::is_regular_file(path)&&!fs::is_symlink(path),"Codec input is missing or not a regular non-symlink file");
 auto bytes=fs::file_size(path);
 guard(bytes>=16&&bytes<=16+size_t(MAX_FRAMES)*Q*2,"Codec input binary size exceeds bounds");
 std::ifstream f(path,std::ios::binary);guard(bool(f),"Cannot open input codec file");
 std::array<unsigned char,16> head{};f.read(reinterpret_cast<char*>(head.data()),16);
 guard(bool(f)&&std::memcmp(head.data(),"Q3OC",4)==0,"Invalid Q3OC magic");
 std::uint32_t version=load_le_u32(head.data()+4), n=load_le_u32(head.data()+8),quantizers=load_le_u32(head.data()+12);
 guard(version==1&&n>=1&&n<=MAX_FRAMES&&quantizers==Q,"Q3OC version/frame/quantizer mismatch");
 guard(bytes==16+std::uint64_t(n)*Q*2,"Q3OC length differs from frame count (trailing or missing bytes)");
 Frames frames(n);
 std::array<unsigned char,Q*2> row{};
 for(auto& frame:frames){f.read(reinterpret_cast<char*>(row.data()),row.size());guard(bool(f),"Truncated Q3OC frame");
  for(int q=0;q<Q;++q){auto c=load_le_u16(row.data()+2*q);guard(c<V,"Codec code ID outside [0,2047]");frame[q]=c;}}
 return frames;
}
void model_preflight(omni::Reader& rd){
 guard(rd.tensors().size()==230&&rd.payload_bytes()==432033154ULL,"Unexpected Code2Wav GGUF count/payload");
 guard(omni::require_string(rd,"general.architecture")=="loraai_qwen3_omni_code2wav_staging"&&
   omni::require_string(rd,"loraai.omni.stage")=="S1-B12A"&&
   omni::require_string(rd,"loraai.omni.component")=="code2wav"&&
   omni::require_string(rd,"loraai.omni.index_sha256")=="03ffe9a54efe9007ceb788c2b036f5c589094918c4582add4e92bc17e1cba838"&&
   omni::require_string(rd,"loraai.omni.config_sha256")=="eab5093d47807aaf894119506b238b2b1cee70d08456e894fee9a012d88f2e0d"&&
   omni::require_uint(rd,"loraai.omni.num_quantizers")==16&&
   omni::require_uint(rd,"loraai.omni.codebook_size")==2048,"Code2Wav GGUF provenance mismatch");
 auto meta=rd.metadata().find("loraai.omni.inference_ready");
 guard(meta!=rd.metadata().end()&&meta->second.type==7&&!meta->second.flag,"Model staging flag invalid");
 auto& t=rd.get_tensor(NAME);
 guard(t.type==omni::GGUF_BF16&&t.dims==std::vector<std::uint64_t>({H,Q*V})&&t.bytes==2ULL*H*Q*V,"BF16 codebook tensor dimensions mismatch");
}
void require_finite(const Vec& v){for(float f:v)guard(std::isfinite(f),"Nonfinite embedding value");}
Vec bf16_embedding_rows(omni::Reader& rd,const Frames& frames){
 const auto& tensor=rd.get_tensor(NAME);Vec data(size_t(Q)*frames.size()*H);
 for(size_t f=0;f<frames.size();++f){for(int q=0;q<Q;++q){
   auto row=std::uint64_t(q)*V+frames[f][q];
   auto raw=rd.read_region(tensor.offset+row*2*H,2*H);
   guard(raw.size()==2*H,"Short BF16 Code2Wav row");
   float* dest=data.data()+(size_t(q)*frames.size()+f)*H;
   for(int j=0;j<H;++j){auto bits=std::uint32_t(load_le_u16(raw.data()+2*j))<<16;std::memcpy(&dest[j],&bits,4);}
 }}require_finite(data);return data;
}
Vec ggml_mean_all(const Vec& values,int frames,int threads){
 guard(frames>=1&&frames<=MAX_FRAMES&&values.size()==size_t(Q)*frames*H,"Embedding graph shape mismatch");
 ggml_init_params p{};p.mem_size=size_t(16)*1024*1024+values.size()*sizeof(float)*3;p.no_alloc=false;
 ggml_context*ctx=ggml_init(p);guard(ctx!=nullptr,"ggml_init failed");
 struct Cleanup{ggml_context* ctx;~Cleanup(){ggml_free(ctx);}} cleanup{ctx};
 ggml_tensor*sum=nullptr;
 for(int q=0;q<Q;++q){auto* t=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,H,frames);
   guard(t!=nullptr,"Cannot allocate GGML embedding tensor");
   std::memcpy(ggml_get_data(t),values.data()+size_t(q)*frames*H,size_t(frames)*H*4);
   sum=sum?ggml_add(ctx,sum,t):t;
 }
 auto*avg=ggml_scale(ctx,sum,1.0f/Q);guard(avg!=nullptr,"GGML scale failed");
 auto*graph=ggml_new_graph(ctx);guard(graph!=nullptr,"GGML graph failed");
 ggml_build_forward_expand(graph,avg);
 guard(ggml_graph_compute_with_ctx(ctx,graph,threads)==GGML_STATUS_SUCCESS,"GGML embedding execution failed");
 Vec result(size_t(frames)*H);std::memcpy(result.data(),ggml_get_data(avg),result.size()*4);
 require_finite(result);return result;
}

}
