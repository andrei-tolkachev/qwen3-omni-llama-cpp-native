#!/usr/bin/env python3
"""Exercise the public PCM client against a synthetic HTTP/SSE server, without weights."""
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

script = Path(__file__).resolve().parents[1] / 'scripts/voice_client.py'


def check(case):
    cancelled = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path.endswith('/cancel'):
                cancelled.append(body['request_id'])
                self.send_response(200)
                self.end_headers()
                return
            assert body['audio_protocol'] == 2 and body['stream']
            generation = body['generation_id']
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            def emit(event, data):
                self.wfile.write(('event: '+event+'\ndata: '+json.dumps(data)+'\n\n').encode())
                self.wfile.flush()
            emit('start', {'generation_id': generation})
            emit('delta', {'text': 'Hello'})
            if case == 'error':
                emit('error', {'error': 'synthetic_failure'})
                return
            for index in range(2):
                chunk = {'generation_id': generation, 'sequence': index,
                         'sample_offset': index * 2, 'format': 'pcm_s16le',
                         'sample_rate': 24000, 'channels': 1,
                         'data': base64.b64encode(b'\x01\x00\x02\x00').decode()}
                # A late packet from a cancelled generation must not reach output.
                emit('audio_chunk', {**chunk, 'generation_id': 'old-generation'})
                if case == 'gap' and index == 1:
                    chunk['sample_offset'] = 3
                emit('audio_chunk', chunk)
            if case == 'eof':
                return
            emit('done', {'generation_id': generation, 'finish_reason': 'stop',
                          'audio_finish_reason': 'stop', 'audio': {'samples': 4, 'chunks': 2}})

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'result.pcm'
            result = subprocess.run([sys.executable, str(script), '--url', f'http://127.0.0.1:{server.server_port}',
                                     '--no-play', '--save', str(target)], capture_output=True, text=True, timeout=15)
            if case == 'ok':
                assert result.returncode == 0, result.stderr
                assert target.read_bytes() == b'\x01\x00\x02\x00' * 2
                assert result.stderr == 'Hello\n'
                assert not cancelled
            else:
                assert result.returncode != 0 and not target.exists(), result.stderr
                assert len(cancelled) == 1
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


for case in ['ok', 'gap', 'eof', 'error']:
    check(case)
print('PASS: PCM playback transport, stale generation, sequence gap, incomplete EOF and error cleanup')
