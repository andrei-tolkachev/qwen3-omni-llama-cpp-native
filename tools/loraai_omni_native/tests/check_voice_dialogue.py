#!/usr/bin/env python3
"""Check multi-turn voice, empty-text errors and recovery against a running server."""
import argparse, base64, json, os, urllib.request, uuid
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--url',default='http://127.0.0.1:8081')
p.add_argument('--wav',type=Path,required=True)
p.add_argument('--turns',type=int,default=5)
a=p.parse_args()
if not 3<=a.turns<=6:p.error('--turns must be between 3 and 6')
audio=base64.b64encode(a.wav.read_bytes()).decode()
client=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def voice(messages,**extra):
 body={'request_id':uuid.uuid4().hex,'messages':messages,'stream':True,'audio_protocol':2,'temperature':0,'max_tokens':96,'max_audio_frames':200,'chat_template_kwargs':{'enable_thinking':False},**extra}
 headers={'Content-Type':'application/json'}
 if os.environ.get('OMNI_API_KEY'):headers['Authorization']='Bearer '+os.environ['OMNI_API_KEY']
 req=urllib.request.Request(a.url.rstrip('/')+'/v1/omni/voice',data=json.dumps(body).encode(),headers=headers)
 events=[];event='';data=[]
 with client.open(req,timeout=600) as response:
  for raw in response:
   line=raw.decode().rstrip('\r\n')
   if line.startswith('event:'):event=line[6:].strip()
   elif line.startswith('data:'):data.append(line[5:].strip())
   elif not line and data:events.append((event,json.loads('\n'.join(data))));data=[]
 return events
messages=[{'role':'system','content':'Answer the speaker briefly in their language.'}]
for i in range(a.turns):
 messages.append({'role':'user','content':[{'type':'input_audio','input_audio':{'data':audio,'format':'wav'}},{'type':'text','text':'Respond to the speaker.'}]})
 events=voice(messages);assert events[-1][0]=='done',events[-1]
 result=events[-1][1];text=result['text'];assert text.strip() and text.strip() not in ('user','assistant'),repr(text)
 assert ''.join(v['text'] for k,v in events if k=='delta')==text
 assert next(i for i,(k,_) in enumerate(events) if k=='delta')<next(i for i,(k,_) in enumerate(events) if k=='audio_chunk')
 generation=events[0][1]['generation_id'];offset=0;parts=[]
 for sequence,chunk in enumerate(v for k,v in events if k=='audio_chunk'):
  assert chunk['generation_id']==generation and chunk['sequence']==sequence and chunk['sample_offset']==offset
  data=base64.b64decode(chunk['data']);parts.append(data);offset+=len(data)//2
 assert result['generation_id']==generation and offset>0
 assert result['audio']['samples']==offset and result['audio']['chunks']==len(parts)
 assert result['audio']['sample_rate']==24000 and result['audio_finish_reason']=='stop'
 assert result['speech_backend']=='gpu',result['speech_backend']
 messages.append({'role':'assistant','content':text});print(json.dumps({'turn':i+1,'text':text,'gpu':True,'multimodal_rows':result['multimodal_rows']}),flush=True)
# A valid zero-character grammar deterministically forces an empty completion.
failed=voice([{'role':'user','content':'Say hello.'}],grammar='root ::= ""')
assert failed[-1]==('error',{'error':'voice_text_empty'}),failed[-1]
recovered=voice([{'role':'user','content':'Say hello in one short sentence.'}])
assert recovered[-1][0]=='done' and recovered[-1][1]['audio']['samples']>0,recovered[-1]
print(json.dumps({'empty_text_error':True,'next_generation_recovers':True,'turns':a.turns}),flush=True)
