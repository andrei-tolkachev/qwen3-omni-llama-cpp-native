#!/usr/bin/env python3
"""Lossless BF16 speech conversion; Python standard library, no GPU or PyTorch.

Historical staging metadata is retained for CPU reference compatibility.
These files are executable by omni-server, not an unmodified llama-server.
"""
import argparse, hashlib, json, math, os, struct
from pathlib import Path


def string(value):
    data=value.encode();return struct.pack('<Q',len(data))+data


def convert(hf, out, dry_run=False):
    schema=json.loads(Path(__file__).with_name('speech_schema.json').read_text())
    index=json.loads((hf/'model.safetensors.index.json').read_text())['weight_map']
    for filename,key in [('config.json','loraai.omni.config_sha256'),('model.safetensors.index.json','loraai.omni.index_sha256')]:
        digest=hashlib.sha256((hf/filename).read_bytes()).hexdigest()
        for component in schema.values():
            assert component['metadata'][key][1]==digest,'Unqualified model config/index: '+filename
    headers={}
    def source(name, shape):
        shard=index[name]
        path=(hf/shard).resolve()
        assert path.parent==hf.resolve(),'Shard outside model directory'
        if shard not in headers:
            with path.open('rb') as f:
                length=struct.unpack('<Q',f.read(8))[0];assert 2<=length<=200*1024*1024
                headers[shard]=(json.loads(f.read(length)),length+8)
        header,start=headers[shard];tensor=header[name]
        assert tensor['dtype']=='BF16' and tensor['shape']==shape, name
        a,b=tensor['data_offsets'];assert 0<=a<b and b-a==2*math.prod(shape) and start+b<=path.stat().st_size,name
        return path,start+a,b-a
    plans=[]
    for filename,component in schema.items():
        entries=[];offset=0
        for tensor in component['tensors']:
            name,shape=tensor['name'],tensor['shape']
            if '.experts_packed.' in name:
                assert shape[0]==128
                sources=[source(name.replace('.experts_packed.',f'.experts.{i}.'),shape[1:]) for i in range(128)]
            else:sources=[source(name,shape)]
            offset=(offset+31)&~31
            entries.append((name,shape,offset,sources));offset+=sum(s[2] for s in sources)
        plans.append((filename,component,entries,offset))
    if dry_run:
        print(json.dumps({'ok':True,'components':len(plans),'tensors':sum(len(p[2]) for p in plans),'bytes':sum(p[3] for p in plans)}));return
    out.mkdir(parents=True,exist_ok=True)
    for filename,component,entries,total in plans:
        target=out/filename;partial=out/(filename+'.partial')
        if target.exists() or partial.exists():raise FileExistsError(target)
        meta=component['metadata']
        header=b'GGUF'+struct.pack('<IQQ',3,len(entries),len(meta))
        for k,(typ,v) in meta.items():
            header+=string(k)+struct.pack('<I',typ)+(string(v) if typ==8 else struct.pack({4:'<I',7:'<?',10:'<Q'}[typ],v))
        for name,shape,offset,sources in entries:
            header+=string(name)+struct.pack('<I',len(shape))+struct.pack('<'+'Q'*len(shape),*reversed(shape))+struct.pack('<IQ',30,offset)
        start=(len(header)+31)&~31
        try:
            with partial.open('xb') as f:
                f.write(header);f.write(b'\0'*(start-len(header)))
                for name,shape,offset,sources in entries:
                    f.write(b'\0'*(start+offset-f.tell()))
                    for path,pos,size in sources:
                        with path.open('rb') as src:
                            src.seek(pos)
                            while size:
                                chunk=src.read(min(size,8*1024*1024))
                                if not chunk:raise EOFError(name)
                                f.write(chunk);size-=len(chunk)
                f.flush();os.fsync(f.fileno())
            partial.rename(target)
        except BaseException:
            partial.unlink(missing_ok=True);raise
        print(filename,total,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hf',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    convert(a.hf,a.out,a.dry_run)
