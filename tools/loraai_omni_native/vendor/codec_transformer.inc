// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andrei Tolkachev
// Derived from the user-provided, BF16-verified LoraAI kernels. See provenance.json.
// LoraAI S1-B13B: Variable-length Code2Wav PRE-TRANSFORMER BF16.
// 1..8 dense Qwen3 layers, 1..512 distinct input temporal frames; sliding 72 attention.
// BF16 weights -> GGML F32 matrix multiply. Scalar C++ RMS/RoPE/softmax/SwiGLU/LayerScale.
// No downstream upsampling/convolution/WAV, no verified Talker or Thinker bridge.
#include "reader.inc"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <array>
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
constexpr int H=1024, NH=16, HD=64, MID=3072, WINDOW=72, MAX_LAYERS=8, MAX_FRAMES=512;
constexpr const char* PREFIX="code2wav.pre_transformer.";
using Vec=std::vector<float>;
using Frames=std::vector<Vec>;
struct Args{fs::path model="omni-experimental/code2wav-staging-bf16.gguf";
 fs::path input,out;int layers=8,threads=2,expected_frames=0;double theta=10000;bool self_test=false;};
void demand(bool ok,const std::string&why){if(!ok)throw omni::Failure(why);}
Args parse(int argc,char**argv){Args a;
 for(int i=1;i<argc;++i){std::string k=argv[i];
  if(k=="--self-test")a.self_test=true;
  else if(k=="--model"&&i+1<argc)a.model=argv[++i];
  else if(k=="--input-f32"&&i+1<argc)a.input=argv[++i];
  else if(k=="--out"&&i+1<argc)a.out=argv[++i];
  else if(k=="--layers"&&i+1<argc)a.layers=std::stoi(argv[++i]);
  else if(k=="--expected-frames"&&i+1<argc)a.expected_frames=std::stoi(argv[++i]);
  else if(k=="--theta"&&i+1<argc)a.theta=std::stod(argv[++i]);
  else if(k=="--threads"&&i+1<argc)a.threads=std::stoi(argv[++i]);
  else throw omni::Failure("S1-B13B unknown/incomplete option: "+k);
 }
 demand(a.layers>=1&&a.layers<=MAX_LAYERS,"layer count must be 1..8");
 demand(a.threads>=1&&a.threads<=32,"threads must be 1..32");
 demand(a.expected_frames>=0&&a.expected_frames<=MAX_FRAMES,"expected frames must be 0..512");
 demand(std::isfinite(a.theta)&&a.theta>=1000&&a.theta<=1e7,"invalid RoPE theta");
 if(!a.self_test)demand(!a.input.empty()&&!a.out.empty(),"--input-f32 and --out required");
 return a;
}
void finite(const Vec&v){for(float x:v)demand(std::isfinite(x),"NaN/Inf in pre-transformer");}
Vec get(omni::Reader&r,const std::string&name,const std::vector<std::uint64_t>&dims){
 const auto&t=r.get_tensor(name);
 demand(t.type==omni::GGUF_BF16&&t.dims==dims,"Code2Wav pre-transformer tensor shape/type mismatch: "+name);
 auto raw=r.read_region(t.offset,static_cast<std::size_t>(t.bytes));
 demand(raw.size()%2==0,"truncated BF16 tensor");Vec out(raw.size()/2);
 for(std::size_t i=0;i<out.size();++i){
  const std::uint16_t b=std::uint16_t(raw[2*i])|(std::uint16_t(raw[2*i+1])<<8);
  const std::uint32_t bits=std::uint32_t(b)<<16;
  std::memcpy(&out[i],&bits,sizeof(float));
 }
 finite(out);return out;
}
Vec linear(const Vec&w,const Vec&x,int ni,int no,int threads){
 demand(x.size()==std::size_t(ni)&&w.size()==std::size_t(ni)*no,"GGML linear geometry mismatch");
 ggml_init_params p{};p.mem_size=4*(w.size()+x.size()+no)+8*1024*1024;p.no_alloc=false;
 auto*ctx=ggml_init(p);demand(ctx!=nullptr,"GGML init failure");
 struct Guard{ggml_context*c;~Guard(){ggml_free(c);}}guard{ctx};
 auto*tw=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,ni,no);
 auto*tx=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,ni,1);
 demand(tw&&tx,"GGML allocate failed");
 std::memcpy(ggml_get_data(tw),w.data(),w.size()*4);
 std::memcpy(ggml_get_data(tx),x.data(),x.size()*4);
 auto*y=ggml_mul_mat(ctx,tw,tx);auto*g=ggml_new_graph(ctx);
 ggml_build_forward_expand(g,y);
 demand(ggml_graph_compute_with_ctx(ctx,g,threads)==GGML_STATUS_SUCCESS,"GGML compute failed");
 Vec out(no);std::memcpy(out.data(),ggml_get_data(y),no*4);finite(out);return out;
}
Vec norm(const Vec&x,const Vec&gain){
 demand(x.size()==H&&gain.size()==H,"RMSNorm geometry invalid");
 double sum=0;for(float v:x)sum+=double(v)*v;
 double inv=1/std::sqrt(sum/H+1e-5);Vec out(H);
 for(int i=0;i<H;++i)out[i]=float(double(x[i])*gain[i]*inv);
 finite(out);return out;
}
void rope(Vec&v,int position,double theta){
 demand(v.size()==H&&position>=0,"RoPE input invalid");
 for(int head=0;head<NH;++head)for(int j=0;j<HD/2;++j){
  const double angle=position/std::pow(theta,2.0*j/HD),c=std::cos(angle),s=std::sin(angle);
  int first=head*HD+j,second=first+HD/2;
  const float a=v[first],b=v[second];
  v[first]=float(a*c-b*s);v[second]=float(b*c+a*s);
 }finite(v);
}
Vec attention(const Vec&q,const std::vector<Vec>&keys,const std::vector<Vec>&values){
 demand(q.size()==H&&keys.size()==values.size()&&!keys.empty()&&keys.size()<=WINDOW,"invalid causal cache");
 Vec out(H,0);
 for(int head=0;head<NH;++head){
  std::vector<double>scores(keys.size());double top=-std::numeric_limits<double>::infinity();
  for(std::size_t t=0;t<keys.size();++t){
   demand(keys[t].size()==H&&values[t].size()==H,"KV cache layer mismatch");
   double dot=0;for(int j=0;j<HD;++j)dot+=double(q[head*HD+j])*keys[t][head*HD+j];
   scores[t]=dot/std::sqrt(double(HD));top=std::max(top,scores[t]);
  }
  double den=0;for(auto &v:scores){v=std::exp(v-top);den+=v;}
  for(std::size_t t=0;t<keys.size();++t){
   double p=scores[t]/den;
   for(int j=0;j<HD;++j)out[head*HD+j]+=float(p*values[t][head*HD+j]);
  }
 }finite(out);return out;
}
Vec swiglu(const Vec&gate,const Vec&up){
 demand(gate.size()==MID&&up.size()==MID,"SwiGLU geometry invalid");Vec result(MID);
 for(int i=0;i<MID;++i){double x=gate[i];result[i]=float((x/(1+std::exp(-x)))*up[i]);}
 finite(result);return result;
}
void scaled_residual(Vec&x,const Vec&v,const Vec&scale){
 demand(x.size()==H&&v.size()==H&&scale.size()==H,"LayerScale geometry invalid");
 for(int i=0;i<H;++i){x[i]=float(double(x[i])+double(v[i])*scale[i]);}
 finite(x);
}
struct Block{
 Vec n1,n2,satt,smlp,q,k,v,o,gate,up,down;
 std::vector<Vec> keys,values;
 Block(omni::Reader&r,int index){
  auto base=std::string(PREFIX)+"layers."+std::to_string(index)+".";
  n1=get(r,base+"input_layernorm.weight",{H});
  n2=get(r,base+"post_attention_layernorm.weight",{H});
  satt=get(r,base+"self_attn_layer_scale.scale",{H});
  smlp=get(r,base+"mlp_layer_scale.scale",{H});
  q=get(r,base+"self_attn.q_proj.weight",{H,H});
  k=get(r,base+"self_attn.k_proj.weight",{H,H});
  v=get(r,base+"self_attn.v_proj.weight",{H,H});
  o=get(r,base+"self_attn.o_proj.weight",{H,H});
  gate=get(r,base+"mlp.gate_proj.weight",{H,MID});
  up=get(r,base+"mlp.up_proj.weight",{H,MID});
  down=get(r,base+"mlp.down_proj.weight",{MID,H});
 }
 Vec forward(const Vec&input,int position,double theta,int threads){
  auto h=norm(input,n1);
  auto qv=linear(q,h,H,H,threads),kv=linear(k,h,H,H,threads),vv=linear(v,h,H,H,threads);
  rope(qv,position,theta);rope(kv,position,theta);
  keys.push_back(std::move(kv));values.push_back(std::move(vv));
  if(keys.size()>WINDOW){keys.erase(keys.begin());values.erase(values.begin());}
  auto att=linear(o,attention(qv,keys,values),H,H,threads);
  Vec y=input;scaled_residual(y,att,satt);
  auto z=norm(y,n2);
  auto f=linear(down,swiglu(linear(gate,z,H,MID,threads),linear(up,z,H,MID,threads)),MID,H,threads);
  scaled_residual(y,f,smlp);return y;
 }
};

}
