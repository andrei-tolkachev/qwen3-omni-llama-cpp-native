#!/usr/bin/env python3
"""Check packed speech GGUF conversion without loading model weights into RAM."""
import argparse,hashlib,json,math,struct
from pathlib import Path


def index(path):
    with path.open('rb') as f:
        def integer(n):
            b=f.read(n)
            if len(b)!=n:raise ValueError('truncated header')
            return int.from_bytes(b,'little')
        def string():
            n=integer(8)
            if n>1048576:raise ValueError('string too large')
            return f.read(n).decode()
        assert f.read(4)==b'GGUF' and integer(4)==3
        count,kv=integer(8),integer(8);assert count<200000 and kv<4096
        meta={}
        for _ in range(kv):
            key,typ=string(),integer(4);assert key not in meta
            meta[key]=string() if typ==8 else integer({4:4,7:1,10:8}[typ])
        tensors={}
        for _ in range(count):
            name=string();rank=integer(4);assert 1<=rank<=4
            dims=[integer(8) for _ in range(rank)];typ,offset=integer(4),integer(8)
            block,size={30:(1,2),12:(256,144),2:(32,18)}[typ]
            assert dims[0]%block==0 and min(dims)>0
            assert name not in tensors
            tensors[name]={'dims':dims,'type':typ,'offset':offset,'bytes':math.prod(dims)//block*size}
        align=meta.get('general.alignment',32);assert align and align<=4096 and align&(align-1)==0
        start=(f.tell()+align-1)//align*align
        end=0
        for t in sorted(tensors.values(),key=lambda t:t['offset']):
            assert t['offset']%align==0 and t['offset']>=end
            end=t['offset']+t['bytes']
        assert start+(end+align-1)//align*align==path.stat().st_size
        return meta,tensors,start


def digest(path,offset,length):
    h=hashlib.sha256()
    with path.open('rb') as f:
        f.seek(offset)
        while length:
            b=f.read(min(length,8*1024*1024));assert b
            h.update(b);length-=len(b)
    return h.hexdigest()


def check(before,after):
    bm,bt,bs=index(before);am,at,astart=index(after)
    assert bt.keys()==at.keys()
    assert all(am[k]==v for k,v in bm.items() if k not in {'general.quantization_version','general.file_type','general.name','general.description'})
    assert am['loraai.omni.quantization']=='Q4_K-gate-up-Q4_0-down-BF16-sensitive-v1'
    counts={};unchanged=0
    for name,b in bt.items():
        a=at[name];assert b['type']==30 and a['dims']==b['dims']
        expected=(12 if a['dims'][0]%256==0 else 2) if '.mlp.experts_packed.' in name else 30
        assert a['type']==expected,name
        counts[str(expected)]=counts.get(str(expected),0)+1
        if expected==30:
            assert digest(before,bs+b['offset'],b['bytes'])==digest(after,astart+a['offset'],a['bytes']),name
            unchanged+=1
    assert counts=={'30':357,'12':40,'2':20},counts
    return {'ok':True,'quantized_experts':60,'unchanged_bf16_tensors':unchanged,'types':counts,'source_bytes':before.stat().st_size,'output_bytes':after.stat().st_size}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bf16',type=Path);p.add_argument('q4',type=Path);a=p.parse_args()
    print(json.dumps(check(a.bf16,a.q4)))
