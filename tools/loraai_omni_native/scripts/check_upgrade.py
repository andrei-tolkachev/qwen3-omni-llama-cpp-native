#!/usr/bin/env python3
"""Run native numerical regression against retained laboratory evidence, offline."""
import argparse,array,json,math,subprocess,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--models',type=Path,required=True);p.add_argument('--build',type=Path,default=root/'build');a=p.parse_args()
f=root/'tests/fixtures';models=a.models/'omni-experimental'
for name in ('omni-duplex-test','omni-voice-capture-test'):
 test=a.build/name
 if not test.is_file():
  raise SystemExit('Build '+name+' with the selected llama checkout before qualification')
 subprocess.run([str(test)],check=True)
with tempfile.TemporaryDirectory(prefix='lora-omni-upgrade-') as temporary:
 out=Path(temporary)
 subprocess.run([str(a.build/'omni-generate'),'--talker',str(models/'talker-experts-packed-bf16-staging.gguf'),'--predictor',str(models/'code-predictor-staging-bf16.gguf'),'--prefill',str(f/'prefill.f32'),'--trailing',str(f/'trailing.f32'),'--pad',str(f/'pad.f32'),'--max-frames','2','--out',str(out/'codes.q3oc')],check=True)
 if (out/'codes.q3oc').read_bytes()!=(f/'expected-two-frames.q3oc').read_bytes(): raise SystemExit('FAIL: recurrent Talker/Predictor codes changed')
 subprocess.run([str(a.build/'omni-decode'),str(models/'code2wav-staging-bf16.gguf'),str(f/'decoder.q3oc'),str(out/'audio.wav')],check=True)
 got=array.array('f',(out/'audio.wav.f32').read_bytes());want=array.array('f',(f/'decoder-reference.f32').read_bytes())
 if len(got)!=len(want) or not all(math.isfinite(x) for x in got): raise SystemExit('FAIL: decoder shape/finite values')
 error=max(abs(x-y) for x,y in zip(got,want))
 if error>1e-4: raise SystemExit(f'FAIL: decoder error {error}')
 print(json.dumps({'ok':True,'exact_recurrent_frames':2,'decoder_samples':len(got),'decoder_max_abs_error':error}))
