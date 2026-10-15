#!/usr/bin/env python3
"""Numerical waveform and seam check; requires the selected GPU backend."""
import argparse,array,json,math,struct,subprocess,tempfile
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True,type=Path);p.add_argument('--decoder',required=True,type=Path);a=p.parse_args()
fixture=Path(__file__).parent/'fixtures/decoder.q3oc';results=[]
with tempfile.TemporaryDirectory(prefix='omni-codec-stream-') as directory:
 work=Path(directory);blob=fixture.read_bytes();codes=(blob[16:]*16)[:112*32]
 longer=work/'long.q3oc';longer.write_bytes(b'Q3OC'+struct.pack('<III',1,len(codes)//32,16)+codes)
 for name,source,chunks in [('reference',fixture,[None,1,3,16]),('long',longer,[None,3,16])]:
  reference=None;count=struct.unpack_from('<I',source.read_bytes(),8)[0]
  for chunk in chunks:
   path=work/f'{name}-{chunk}.wav';subprocess.run([str(a.decoder),str(a.model),str(source),str(path),*([str(chunk)] if chunk else [])],check=True)
   got=array.array('f',Path(str(path)+'.f32').read_bytes())
   if reference is None:reference=got
   assert len(got)==len(reference) and all(math.isfinite(v) for v in got)
   maximum=max(abs(x-y) for x,y in zip(got,reference));rms=math.sqrt(sum((x-y)**2 for x,y in zip(got,reference))/len(got));signal=math.sqrt(sum(x*x for x in reference)/len(reference));snr=20*math.log10(signal/max(rms,1e-12))
   borders=range(chunk,count,chunk) if chunk else []
   seam=max([abs(got[i]-reference[i]) for frame in borders for i in range(max(0,frame*1920-555-32),min(len(got),frame*1920-555+32))]+[0])
   row=dict(fixture=name,chunk=chunk,samples=len(got),max_abs_error=maximum,snr_db=snr,boundary_max_abs_error=seam);results.append(row)
   assert maximum<.005 and snr>55,row
print(json.dumps({'ok':True,'checks':results},indent=2))
