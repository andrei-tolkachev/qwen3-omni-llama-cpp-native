#include "audio_join.h"
#include <cassert>
#include <cmath>
#include <cstdio>
using namespace lora_omni;
int main() {
    Vector out;
    AudioJoin join([&](const Vector & x){out.insert(out.end(), x.begin(), x.end());});
    join.next_phrase(); join.push(Vector(1000,1.f)); join.push(Vector(1000,1.f));
    join.next_phrase(); join.push(Vector(1000,-1.f)); join.finish();
    assert(out.size()==2760);
    for(size_t i=1;i<out.size();++i)assert(std::abs(out[i]-out[i-1])<.009f);
    out.clear();
    Vector source(10000);
    for(size_t i=0;i<source.size();++i)source[i]=std::sin(float(i)*.1f);
    AudioJoin continuous([&](const Vector & x){out.insert(out.end(),x.begin(),x.end());});
    continuous.next_phrase();
    for(size_t i=0;i<source.size();i+=257)continuous.push(Vector(source.begin()+i,source.begin()+std::min(source.size(),i+257)));
    continuous.finish();assert(out==source);
    std::puts("PASS: phrase crossfade suppresses boundary step without added silence; within-phrase samples unchanged");
}
