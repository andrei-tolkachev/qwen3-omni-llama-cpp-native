// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andrei Tolkachev
// Derived from the user-provided, BF16-verified LoraAI kernels. See provenance.json.
// Qwen3-Omni S1-B13D2: full FIRST DAC cascade for variable frame-length controlled codec IDs.
// 28x1536 B13D1 stem -> ConvTranspose1d(k16,s8,crop8+8) 216x768 -> 3 residual units.
// Only isolated GGUF Bf16 + GGML; this does NOT generate speech or output WAV.
// Reuses installed, accepted S1-B13D1 GGML kernels; no modification to their implementation.
#include "codec_stem.inc"
#include <array>
#include <numeric>
namespace {
constexpr int C_IN=1536, C_OUT=768, C_K=16, C_STRIDE=8, C_CROP=C_K-C_STRIDE;
constexpr std::array<int,3> DILATIONS{{1,3,9}};
struct CArgs {fs::path model=MODEL_DEFAULT, input,out;int expected_frames=0,threads=2;bool self_test=false;};
CArgs parse_c(int argc,char**argv){CArgs a;
 for(int i=1;i<argc;++i){std::string s=argv[i];
  if(s=="--self-test")a.self_test=true;
  else if(s=="--model"&&i+1<argc)a.model=argv[++i];
  else if(s=="--input-f32"&&i+1<argc)a.input=argv[++i];
  else if(s=="--out"&&i+1<argc)a.out=argv[++i];
  else if(s=="--expected-frames"&&i+1<argc)a.expected_frames=std::stoi(argv[++i]);
  else if(s=="--threads"&&i+1<argc)a.threads=std::stoi(argv[++i]);
  else throw omni::Failure("unknown/incomplete cascade-0 argument: "+s);
 }
 check(a.threads>=1&&a.threads<=32,"cascade-0 threads must be 1..32");
 if(!a.self_test){check(!a.input.empty()&&!a.out.empty(),"--input-f32 and --out required");
  check(a.expected_frames>=4 && a.expected_frames<=MAX_DAC_FRAMES && a.expected_frames%4==0,
      "first DAC cascade requires all 4*N stem frames, 1<=N<=512");}
 else check(a.input.empty()&&a.out.empty(),"self-test does not accept input/output files");
 return a;
}
std::vector<std::string> suffixes(){std::vector<std::string>s={"",".activated.f32",".raw.f32",".transpose.f32",".json"};
 for(int u=0;u<3;++u)for(auto stage:{"act1","conv1","act2","conv2","residual"})
  s.push_back(".u"+std::to_string(u)+"."+stage+".f32");
 return s;
}
int output_frames(int n,int stride,int kernel){
 check(n>1&&stride>0&&kernel>=stride,"invalid DAC transpose geometry");
 const int raw=(n-1)*stride+kernel,crop=kernel-stride;
 check(raw>2*crop,"cropping would eliminate transposed output");
 return raw-2*crop;
}
struct Trans {Vec raw,cropped;};
Trans transpose(const Vec& x,const Vec& w,const Vec& b,int frames,int ci,int co,int k,int stride,int threads){
 check(x.size()==size_t(frames)*ci&&w.size()==size_t(ci)*co*k&&b.size()==size_t(co),
       "transposed convolution tensor size mismatch");
 const int crop=k-stride,raw_frames=(frames-1)*stride+k,nt=output_frames(frames,stride,k);
 // Original PyTorch ConvTranspose weight is [input_channel,output_channel,kernel_tap].
 // GGML matrix W ne0=input_channel; ne1=(output_channel*k+tap), transposed from HF layout.
 Vec reordered(w.size());
 for(int ic=0;ic<ci;++ic)for(int oc=0;oc<co;++oc)for(int tap=0;tap<k;++tap)
  reordered[(size_t(oc)*k+tap)*ci+ic]=w[(size_t(ic)*co+oc)*k+tap];
 auto folded=gemm(reordered,x,ci,co*k,frames,threads);
 Trans out;out.raw.resize(size_t(raw_frames)*co);
 for(int t=0;t<raw_frames;++t)std::copy(b.begin(),b.end(),out.raw.begin()+size_t(t)*co);
 for(int t=0;t<frames;++t)for(int oc=0;oc<co;++oc)for(int tap=0;tap<k;++tap)
  out.raw[size_t(t*stride+tap)*co+oc]+=folded[(size_t(t)*co+oc)*k+tap];
 out.cropped.assign(out.raw.begin()+size_t(crop)*co,out.raw.begin()+size_t(crop+nt)*co);
 finite(out.raw);finite(out.cropped);return out;
}
Vec dilated_conv(const Vec&x,const Vec&w,const Vec&b,int t,int ch,int k,int dilation,int threads){
 check(t>0&&ch>0&&k>0&&dilation>0&&x.size()==size_t(t)*ch&&
       w.size()==size_t(ch)*ch*k&&b.size()==size_t(ch),"DAC dilated Conv1d geometry invalid");
 Vec cols(size_t(t)*ch*k,0.f);
 for(int ti=0;ti<t;++ti)for(int ic=0;ic<ch;++ic)for(int tap=0;tap<k;++tap){
  const int at=ti-(k-1-tap)*dilation;
  if(at>=0)cols[(size_t(ti)*ch+ic)*k+tap]=x[size_t(at)*ch+ic];
 }
 auto y=gemm(w,cols,ch*k,ch,t,threads);
 for(int ti=0;ti<t;++ti)for(int oc=0;oc<ch;++oc)y[size_t(ti)*ch+oc]+=b[oc];
 finite(y);return y;
}
struct Unit {Vec act1,conv1,act2,conv2,residual;};
struct UnitWeights{Vec a1,b1,w1,c1,a2,b2,w2,c2;};
UnitWeights read_unit(omni::Reader&r,int index,int ch,int stage=1){
 check(index>=0&&index<3,"invalid DAC residual unit index");
 std::string base="code2wav.decoder."+std::to_string(stage)+".block."+std::to_string(index+2)+".";
 return {get(r,base+"act1.alpha",{std::uint64_t(ch)}),
         get(r,base+"act1.beta",{std::uint64_t(ch)}),
         get(r,base+"conv1.conv.weight",{7,std::uint64_t(ch),std::uint64_t(ch)}),
         get(r,base+"conv1.conv.bias",{std::uint64_t(ch)}),
         get(r,base+"act2.alpha",{std::uint64_t(ch)}),
         get(r,base+"act2.beta",{std::uint64_t(ch)}),
         get(r,base+"conv2.conv.weight",{1,std::uint64_t(ch),std::uint64_t(ch)}),
         get(r,base+"conv2.conv.bias",{std::uint64_t(ch)})};
}
Unit unit(const Vec&x,const UnitWeights&w,int frames,int channels,int dilation,int threads){
 Unit u;
 u.act1=snake(x,w.a1,w.b1,frames,channels);
 u.conv1=dilated_conv(u.act1,w.w1,w.c1,frames,channels,7,dilation,threads);
 u.act2=snake(u.conv1,w.a2,w.b2,frames,channels);
 u.conv2=dilated_conv(u.act2,w.w2,w.c2,frames,channels,1,1,threads);
 u.residual.resize(x.size());
 for(size_t i=0;i<x.size();++i)u.residual[i]=x[i]+u.conv2[i];
 finite(u.residual);return u;
}

}
