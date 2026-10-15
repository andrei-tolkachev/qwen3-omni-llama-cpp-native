// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andrei Tolkachev
// Derived from the user-provided, BF16-verified LoraAI kernels. See provenance.json.
// LoraAI Qwen3 Omni native Code2Wav upsample microtest.
// Two stages: ConvTranspose1d(stride=2,kernel=2) -> ConvNeXt1D.
// GGML F32 matrix products on original BF16 GGUF weights. Scalar C++ causal DW,
// LayerNorm and GELU for parity work; NOT audio inference or a complete Code2Wav.
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
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
namespace fs=std::filesystem;
namespace {
constexpr int H=1024, K=2, DW=7, EXP=4096, MAX_FRAMES=512;
using Vec=std::vector<float>;
struct Args {
 fs::path model="omni-experimental/code2wav-staging-bf16.gguf";
 fs::path input, output; int stages=2,threads=2,expected_frames=0; bool self_test=false;
};
void require(bool v,const std::string&why){if(!v)throw omni::Failure(why);}
Args parse(int argc,char**argv){Args a;for(int i=1;i<argc;++i){std::string k=argv[i];
 if(k=="--self-test")a.self_test=true;
 else if(k=="--model"&&i+1<argc)a.model=argv[++i];
 else if(k=="--input-f32"&&i+1<argc)a.input=argv[++i];
 else if(k=="--out"&&i+1<argc)a.output=argv[++i];
 else if(k=="--stages"&&i+1<argc)a.stages=std::stoi(argv[++i]);
 else if(k=="--threads"&&i+1<argc)a.threads=std::stoi(argv[++i]);
 else if(k=="--expected-frames"&&i+1<argc)a.expected_frames=std::stoi(argv[++i]);
 else throw omni::Failure("unknown/incomplete option: "+k);
 }
 require(a.stages==1||a.stages==2,"stages must be 1 or 2");
 require(a.threads>=1&&a.threads<=32,"threads must be 1..32");
 if(!a.self_test){require(!a.input.empty()&&!a.output.empty(),"--input-f32 and --out required");
 require(a.expected_frames>=1&&a.expected_frames<=MAX_FRAMES,"expected frames must be 1..512");}
 return a;
}
void finite(const Vec&v){for(float x:v)require(std::isfinite(x),"NaN/Inf in Code2Wav upsample");}
Vec get(omni::Reader&r,const std::string&name,const std::vector<std::uint64_t>&dims){
 const auto&t=r.get_tensor(name);
 require(t.type==omni::GGUF_BF16 && t.dims==dims,"bad type/shape for "+name);
 auto raw=r.read_region(t.offset,static_cast<std::size_t>(t.bytes));
 require(raw.size()==t.bytes && (raw.size()%2)==0,"truncated BF16 tensor");
 Vec result(raw.size()/2);
 for(size_t i=0;i<result.size();++i){
  std::uint16_t h=std::uint16_t(raw[2*i])|(std::uint16_t(raw[2*i+1])<<8);
  std::uint32_t bits=std::uint32_t(h)<<16;std::memcpy(&result[i],&bits,4);
 }
 finite(result);return result;
}
// F32 batched GEMM: HF [out,in] (or flattened [in,out*kernel] for transposed conv),
// GGML [in,out], input [in,time], output [out,time].
Vec gemm(const Vec&w,const Vec&x,int ni,int no,int t,int threads){
 require(ni>0&&no>0&&t>0&&x.size()==size_t(ni)*t&&w.size()==size_t(ni)*no,"gemm shape mismatch");
 ggml_init_params p{};p.mem_size=8*1024*1024+(w.size()+x.size()+size_t(no)*t)*4;
 p.no_alloc=false;
 ggml_context *ctx=ggml_init(p);require(ctx!=nullptr,"GGML init failed");
 struct Guard{ggml_context*c;~Guard(){ggml_free(c);}}guard{ctx};
 auto*tw=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,ni,no);
 auto*tx=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,ni,t);
 require(tw&&tx,"GGML alloc failed");
 std::memcpy(ggml_get_data(tw),w.data(),w.size()*4);
 std::memcpy(ggml_get_data(tx),x.data(),x.size()*4);
 auto*yy=ggml_mul_mat(ctx,tw,tx);
 auto*g=ggml_new_graph(ctx);require(g!=nullptr,"GGML graph failed");
 ggml_build_forward_expand(g,yy);
 require(ggml_graph_compute_with_ctx(ctx,g,threads)==GGML_STATUS_SUCCESS,"GGML GEMM failure");
 Vec y(size_t(no)*t);std::memcpy(y.data(),ggml_get_data(yy),y.size()*4);finite(y);return y;
}
// [time][channel] F32 layout, i.e. contiguous channels per frame.
Vec transposed(const Vec&x,const Vec&w,const Vec&bias,int t,int threads){
 require(x.size()==size_t(t)*H && w.size()==size_t(H)*H*K && bias.size()==H,"transposed conv geometry invalid");
 // HF ConvTranspose weights [in,out,k] are NOT laid out [out*k,in]!
 // Reorder to GGML's contiguous dim0=IN before matrix multiply.
 Vec matrix(w.size());
 for(int ic=0;ic<H;++ic)for(int oc=0;oc<H;++oc)for(int k=0;k<K;++k)
  matrix[(oc*K+k)*H+ic]=w[(ic*H+oc)*K+k];
 auto raw=gemm(matrix,x,H,H*K,t,threads);
 Vec out(size_t(t)*2*H);
 for(int frame=0;frame<t;++frame)for(int oc=0;oc<H;++oc)for(int k=0;k<K;++k)
  out[(2*frame+k)*H+oc]=raw[frame*H*K+oc*K+k]+bias[oc];
 finite(out);return out;
}
Vec depthwise(const Vec&x,const Vec&w,const Vec&bias,int t){
 require(x.size()==size_t(t)*H&&w.size()==size_t(H)*DW&&bias.size()==H,"DW geometry invalid");
 Vec out(x.size());
 // PyTorch CausalConv1d pads 6 zeros on the LEFT for kernel=7,stride=1.
 for(int ti=0;ti<t;++ti)for(int c=0;c<H;++c){
  double s=bias[c];for(int k=0;k<DW;++k){int source=ti-(DW-1)+k;
   if(source>=0)s+=double(x[size_t(source)*H+c])*w[c*DW+k];}
  out[size_t(ti)*H+c]=float(s);
 }
 finite(out);return out;
}
Vec layer_norm(const Vec&x,const Vec&weight,const Vec&bias,int t){
 require(x.size()==size_t(t)*H && weight.size()==H && bias.size()==H,"LayerNorm geometry invalid");
 Vec out(x.size());
 for(int ti=0;ti<t;++ti){
  const float *v=x.data()+size_t(ti)*H;
  double mu=0;for(int c=0;c<H;++c)mu+=v[c];mu/=H;
  double var=0;for(int c=0;c<H;++c){auto d=v[c]-mu;var+=d*d;}var/=H;
  double inv=1/std::sqrt(var+1.e-6);
  for(int c=0;c<H;++c)out[size_t(ti)*H+c]=float((v[c]-mu)*inv*weight[c]+bias[c]);
 }
 finite(out);return out;
}
void bias_gelu(Vec&v,const Vec&bias,int t){
 require(v.size()==size_t(t)*EXP&&bias.size()==EXP,"pwconv1 bias geometry invalid");
 for(int ti=0;ti<t;++ti)for(int c=0;c<EXP;++c){
  auto idx=size_t(ti)*EXP+c;double z=double(v[idx])+bias[c];
  v[idx]=float(z*0.5*(1+std::erf(z/std::sqrt(2.))));
 }finite(v);
}
void residual(Vec&out,const Vec&original,const Vec&bias,const Vec&gamma,int t){
 require(out.size()==original.size()&&out.size()==size_t(t)*H&&bias.size()==H&&gamma.size()==H,"ConvNeXt residual shape mismatch");
 for(int ti=0;ti<t;++ti)for(int c=0;c<H;++c){auto idx=size_t(ti)*H+c;
  out[idx]=float(double(original[idx])+double(gamma[c])*(double(out[idx])+bias[c]));
 }finite(out);
}
struct Upsample {
 Vec trans_w,trans_bias,dw_w,dw_bias,norm_w,norm_bias,pw1_w,pw1_bias,pw2_w,pw2_bias,gamma;
 Upsample(omni::Reader&r,int stage){
  std::string base="code2wav.upsample."+std::to_string(stage)+".";
  trans_w=get(r,base+"0.conv.weight",{2,H,H});
  trans_bias=get(r,base+"0.conv.bias",{H});
  dw_w=get(r,base+"1.dwconv.conv.weight",{DW,1,H});
  dw_bias=get(r,base+"1.dwconv.conv.bias",{H});
  norm_w=get(r,base+"1.norm.weight",{H});
  norm_bias=get(r,base+"1.norm.bias",{H});
  pw1_w=get(r,base+"1.pwconv1.weight",{H,EXP});
  pw1_bias=get(r,base+"1.pwconv1.bias",{EXP});
  pw2_w=get(r,base+"1.pwconv2.weight",{EXP,H});
  pw2_bias=get(r,base+"1.pwconv2.bias",{H});
  gamma=get(r,base+"1.gamma",{H});
 }
 Vec forward(const Vec&input,int t,int threads,Vec*trans_output=nullptr){
  Vec x=transposed(input,trans_w,trans_bias,t,threads);int frames=t*2;
  if(trans_output)*trans_output=x;
  auto dw=depthwise(x,dw_w,dw_bias,frames);
  auto norm=layer_norm(dw,norm_w,norm_bias,frames);
  auto pw1=gemm(pw1_w,norm,H,EXP,frames,threads);
  bias_gelu(pw1,pw1_bias,frames);
  auto pw2=gemm(pw2_w,pw1,EXP,H,frames,threads);
  residual(pw2,x,pw2_bias,gamma,frames);
  return pw2;
 }
};

}
