#!/usr/bin/env python3
"""Real-model overlap, deterministic PCM, rollback and fragmented audio checks.
Runs against an isolated OMNI_DUPLEX=1 server; does not change running services.
"""
import argparse
import base64
import hashlib
import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

class Session:
    def __init__(self, url, messages=None, **options):
        self.url = url.rstrip('/')
        self.events = queue.Queue()
        self.log = []
        self.pcm = {}
        self.sequence = 0
        self.id = None
        self.response = self.request('', {'messages': messages or [], 'max_tokens': 256,
                                         'speech_seed': 42, 'temperature': 0, **options}, stream=True)
        self.reader = threading.Thread(target=self.read, daemon=True)
        self.reader.start()
        self.id = self.wait('session')['session_id']

    def request(self, route, body, stream=False):
        headers = {'Content-Type': 'application/json'}
        if os.environ.get('OMNI_API_KEY'):
            headers['Authorization'] = 'Bearer ' + os.environ['OMNI_API_KEY']
        req = urllib.request.Request(self.url + '/v1/omni/duplex' + route,
                                     json.dumps(body).encode(), headers)
        response = urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=300)
        if stream:
            return response
        with response:
            return json.load(response)

    def read(self):
        try:
            event, data = '', []
            for raw in self.response:
                line = raw.decode().rstrip('\r\n')
                if line.startswith('event:'):
                    event = line[6:].strip()
                elif line.startswith('data:'):
                    data.append(line[5:].strip())
                elif not line and data:
                    item = json.loads('\n'.join(data)); data = []
                    item['_time'] = time.monotonic()
                    if event == 'audio_chunk':
                        pcm = base64.b64decode(item['data'], validate=True)
                        state = self.pcm.setdefault(item['generation_id'], {'samples': 0, 'chunks': 0, 'hash': hashlib.sha256()})
                        assert item['sequence'] == state['chunks'], item
                        assert item['sample_offset'] == state['samples'], item
                        assert item['format'] == 'pcm_s16le' and item['sample_rate'] == 24000
                        state['samples'] += len(pcm) // 2; state['chunks'] += 1; state['hash'].update(pcm)
                    if event != 'keepalive':
                        self.log.append((event, {k: v for k, v in item.items() if k != 'data'}))
                        self.events.put((event, item))
        except BaseException as e:
            self.events.put(('error', {'error': repr(e)}))
        finally:
            self.events.put(('closed', {}))

    def wait(self, event, timeout=180, **fields):
        deadline = time.monotonic() + timeout
        while True:
            kind, item = self.events.get(timeout=max(.01, deadline - time.monotonic()))
            if kind in ('error', 'closed'):
                raise AssertionError((kind, item, self.log[-8:]))
            if kind == event and all(item.get(k) == v for k, v in fields.items()):
                return item
            if time.monotonic() >= deadline:
                raise TimeoutError(event)

    def append(self, content, final):
        result = self.request('/input', {'session_id': self.id, 'sequence': self.sequence,
                                        'content': content, 'final': final})
        assert result['accepted_sequence'] == self.sequence
        self.sequence += 1
        return result

    def cancel(self, generation):
        return self.request('/cancel', {'session_id': self.id, 'generation_id': generation})

    def ack(self, generation, samples, final=True):
        return self.request('/ack', {'session_id': self.id, 'generation_id': generation,
                                     'played_samples': samples, 'final':final})

    def close(self):
        self.request('/close', {'session_id': self.id})
        self.reader.join(timeout=30)
        self.response.close()
        assert not self.reader.is_alive()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


def text(value): return [{'type': 'text', 'text': value}]

FIRST = ('Read this paragraph aloud exactly, with no introduction: '
         'The quiet garden was full of birds in the early morning. A warm breeze moved through the trees, '
         'and sunlight fell across the path. We sat beside the fountain and talked about our plans for the day. '
         'After breakfast we would walk to the river, visit the old bridge, and return home before the evening rain.')


def benchmark(url, overlap):
    with Session(url, [{'role': 'system', 'content': 'Follow the user instructions. Reply in plain spoken English.'}]) as s:
        s.append(text(FIRST), True)
        started = s.wait('speech_started')
        s.wait('audio_chunk', generation_id=started['generation_id'])
        if not overlap:
            done = s.wait('speech_done', generation_id=started['generation_id'])
        s.append(text('Remember the number seven. '), False)
        partial = s.wait('input_processed', sequence=1)
        s.append(text('What is that number plus five? Answer in one short sentence.'), True)
        final = s.wait('input_processed', sequence=2, final=True)
        if overlap:
            done = next((v for k, v in s.log if k == 'speech_done' and v['generation_id'] == started['generation_id']), None)
            if done is None:
                done = s.wait('speech_done', generation_id=started['generation_id'])
        s.ack(started['generation_id'], done['samples'])
        second = s.wait('speech_started', sequence=2)
        s.wait('speech_done', generation_id=second['generation_id'])
        state = s.pcm[started['generation_id']]
        assert done['samples'] == state['samples'] and done['chunks'] == state['chunks']
        if overlap:
            assert started['_time'] < partial['_time'] < done['_time'], 'No real Thinker/Talker overlap'
            assert any(k == 'audio_chunk' and v['generation_id'] == started['generation_id'] and
                       v['_time'] > partial['_time'] for k, v in s.log), 'No continuing PCM after new input processing'
        return {'overlap': overlap, 'speech_ms': done['speech_ms'], 'samples': state['samples'],
                'pcm_sha256': state['hash'].hexdigest(), 'text': started['text'],
                'partial_thinker_ms': partial['thinker_ms'], 'final_thinker_ms': final['thinker_ms'],
                'next_text': second['text'], 'rtf': done['speech_ms'] / 1000 / (state['samples'] / 24000),
                'events': s.log}


def cancellation(url, after_synthesis=False):
    with Session(url) as s:
        s.append(text(FIRST), True)
        first = s.wait('speech_started')
        s.wait('audio_chunk', generation_id=first['generation_id'])
        if after_synthesis:
            s.wait('speech_done', generation_id=first['generation_id'])
        s.append(text('Answer only: Everything is ready.'), True)
        ready = s.wait('input_processed', sequence=1, final=True)
        t0 = time.monotonic()
        assert s.cancel(first['generation_id'])['cancelled']
        assert not s.cancel(first['generation_id'])['cancelled']
        corrected = s.wait('input_processed', sequence=1, final=True)
        assert corrected['revision'] > ready['revision']
        assert corrected['generation_id'] != ready['generation_id']
        second = s.wait('speech_started', sequence=1)
        s.wait('speech_done', generation_id=second['generation_id'])
        return {'rollback_recomputed': True, 'recompute_seconds': corrected['_time'] - t0,
                'next_text': second['text']}


def fragmented_audio(url, wav):
    # Each packet is a complete WAV container. Splitting raw PCM without WAV
    # headers is intentionally not supported by the upstream audio frontend.
    import io
    import wave
    with wave.open(str(wav), 'rb') as source:
        params = source.getparams(); frames = source.readframes(params.nframes)
    width = params.sampwidth * params.nchannels
    cut = (params.nframes // 2) * width
    blocks = []
    for raw in (frames[:cut], frames[cut:]):
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as target:
            target.setparams(params); target.writeframes(raw)
        blocks.append({'type': 'input_audio', 'input_audio': {'format': 'wav', 'data': base64.b64encode(buf.getvalue()).decode()}})
    with Session(url, [{'role': 'system', 'content': 'Answer briefly in the language of the speaker.'}]) as s:
        s.append(text(FIRST), True)
        first = s.wait('speech_started'); s.wait('audio_chunk', generation_id=first['generation_id'])
        s.append([blocks[0]], False)
        processed = s.wait('input_processed', sequence=1)
        assert not any(k == 'speech_done' and v['generation_id'] == first['generation_id'] for k, v in s.log), 'Audio input did not overlap'
        s.append([blocks[1], *text('Respond briefly to the audio.')], True)
        final = s.wait('input_processed', sequence=2, final=True)
        done = next((v for k, v in s.log if k == 'speech_done' and v['generation_id'] == first['generation_id']), None)
        if done is None: done = s.wait('speech_done', generation_id=first['generation_id'])
        s.ack(first['generation_id'], done['samples'])
        second = s.wait('speech_started', sequence=2); s.wait('speech_done', generation_id=second['generation_id'])
        return {'audio_fragments': 2, 'overlap': True, 'partial_thinker_ms': processed['thinker_ms'],
                'final_thinker_ms': final['thinker_ms'], 'text': second['text']}


def adaptive_checks(url):
    """Immediate PCM-consumption ACKs test state/compute, not device playback."""
    with Session(url, adaptive_dialogue=True) as s:
        s.append(text(FIRST), True)
        start = s.wait('speech_started')
        generation = start['generation_id']
        first_pcm = s.wait('audio_chunk', generation_id=generation)
        played = first_pcm['sample_offset'] + len(base64.b64decode(first_pcm['data'])) // 2
        s.ack(generation, played, final=False)
        query_result = {}
        def query():
            try:
                query_result['started'] = time.monotonic()
                query_result['response'] = s.request('/query', {'session_id':s.id, 'query':{
                    'messages':[{'role':'user','content':'What is two plus two? Reply with the number only.'}], 'max_tokens':32}})
                query_result['finished'] = time.monotonic()
            except BaseException as error:
                query_result['error'] = repr(error)
        worker = threading.Thread(target=query, daemon=True); worker.start()
        prefixes = 0; pending = []; heard = ''
        while True:
            kind, value = s.events.get(timeout=180)
            if kind in ('error','closed'): raise AssertionError((kind,value))
            if kind == 'audio_chunk':
                played = value['sample_offset'] + len(base64.b64decode(value['data'])) // 2
                s.ack(generation, played, final=False)
            elif kind == 'phrase_boundary':
                pending.append(value)
            for boundary in list(pending):
                if boundary['sample_end'] > played: continue
                snapshot = s.request('/context', {'session_id':s.id})
                current = snapshot['messages'][-1]['content']
                expected = start['text'].encode()[:boundary['text_end']].decode()
                assert current.startswith(expected) and current.startswith(heard) and start['text'].startswith(current)
                heard = current; pending.remove(boundary); prefixes += 1
            if kind == 'speech_done':
                s.ack(generation,value['samples']); break
        worker.join(timeout=30); assert not worker.is_alive() and 'error' not in query_result, query_result
        assert query_result['response']['text'].strip().strip('.') == '4', query_result
        assert query_result['finished'] < value['_time'], 'Query did not finish during speech synthesis'
        assert prefixes > 1
        before = s.request('/context', {'session_id':s.id})['messages']
        utterance = 'The check is running.'
        s.request('/input', {'session_id':s.id, 'sequence':s.sequence, 'final':True,
            'content':text(utterance), 'record_input':False, 'spoken_text':utterance})
        s.sequence += 1
        spoken = s.wait('speech_started'); assert spoken['text'] == utterance
        done = s.wait('speech_done', generation_id=spoken['generation_id'])
        s.ack(spoken['generation_id'],done['samples'])
        after = s.request('/context', {'session_id':s.id})['messages']
        assert after == before + [{'role':'assistant','content':utterance}]
    history = [{'role':'system','content':'Answer the current request.'},
               {'role':'user','content':'Say amber.'},{'role':'assistant','content':'amber'}]
    with Session(url, history, adaptive_dialogue=True) as s:
        s.request('/input', {'session_id':s.id, 'sequence':0, 'final':True,
            'content':text('Say the required color only.'),
            'context':'The required color for this turn is violet.'})
        started=s.wait('speech_started'); assert started['text'].strip().lower().strip('.')=='violet',started
        done=s.wait('speech_done');s.ack(started['generation_id'],done['samples'])
        after=s.request('/context',{'session_id':s.id})['messages']
        assert after[:len(history)]==history and len(after)==len(history)+2
        assert 'required color for this turn' not in json.dumps(after)
    return {'heard_prefixes':prefixes, 'query_during_synthesis':True,
            'verbatim_continuation':True, 'turn_context_not_persisted':True,
            'original_history_unchanged':True, 'playback':'immediate consumption ACKs'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:18081')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--wav', type=Path)
    p.add_argument('--repeats', type=int, default=2)
    a = p.parse_args()
    results = {'benchmark': []}
    # Warm both native paths before comparing scheduling overhead.
    with Session(a.url) as s:
        s.append(text('Say exactly: The system is ready.'), True)
        s.wait('speech_done')
    for i in range(a.repeats):
        for overlap in (False, True):
            result = benchmark(a.url, overlap)
            results['benchmark'].append(result)
            a.output.write_text(json.dumps(results, indent=2))
            print(json.dumps({k: v for k, v in result.items() if k != 'events'}), flush=True)
    hashes = {r['pcm_sha256'] for r in results['benchmark']}
    assert len(hashes) == 1, 'Concurrent speech PCM differs from the same seeded isolated generation'
    results['pcm_bit_identical'] = True
    results['cancellation'] = cancellation(a.url)
    results['cancel_after_synthesis'] = cancellation(a.url, True)
    results['adaptive'] = adaptive_checks(a.url)
    if a.wav:
        results['fragmented_audio'] = fragmented_audio(a.url, a.wav)
    a.output.write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != 'benchmark'}), flush=True)

if __name__ == '__main__': main()
