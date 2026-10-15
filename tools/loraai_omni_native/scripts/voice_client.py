#!/usr/bin/env python3
"""Stream text and protocol-2 PCM from omni-server; Ctrl-C cancels generation."""
import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import uuid


def events(response):
    name, data, size = "", [], 0
    while True:
        raw = response.readline(1_048_577)
        if not raw:
            if data:
                raise ValueError("incomplete SSE event")
            return
        size += len(raw)
        if size > 1_048_576:
            raise ValueError("oversized SSE event")
        line = raw.decode("utf-8").rstrip("\r\n")
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
        elif not line:
            if data:
                yield name, json.loads("\n".join(data))
            name, data, size = "", [], 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    parser.add_argument("--text", default="Say hello briefly.")
    parser.add_argument("--wav", type=Path, help="optional recorded input turn")
    parser.add_argument("--save", type=Path, help="save raw 24 kHz mono S16LE PCM")
    parser.add_argument("--no-play", action="store_true")
    parser.add_argument("--api-key-file", type=Path)
    args = parser.parse_args()
    key = args.api_key_file.read_text().strip() if args.api_key_file else os.environ.get("OMNI_API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    request_id = uuid.uuid4().hex
    generation_id = uuid.uuid4().hex
    content = args.text
    if args.wav:
        content = [{"type": "input_audio", "input_audio": {
            "format": "wav", "data": base64.b64encode(args.wav.read_bytes()).decode()}},
            {"type": "text", "text": args.text}]
    body = {"request_id": request_id, "generation_id": generation_id,
            "stream": True, "audio_protocol": 2,
            "messages": [{"role": "user", "content": content}]}
    # Do not send API keys through environment-configured proxies or redirects.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *unused):
            return None
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def post(route, payload, timeout=120):
        return client.open(urllib.request.Request(args.url.rstrip("/") + route,
                           data=json.dumps(payload).encode(), headers=headers), timeout=timeout)

    player, output, response = None, None, None
    complete, started = False, False
    sequence, offset = 0, 0
    try:
        if args.save:
            output = args.save.open("xb")
        response = post("/v1/omni/voice", body)
        if "text/event-stream" not in response.headers.get("Content-Type", ""):
            raise ValueError("server did not return SSE")
        for event, value in events(response):
            if event == "start":
                if started or value.get("generation_id") != generation_id:
                    raise ValueError("invalid generation start")
                started = True
            elif event == "delta":
                if not started:
                    raise ValueError("delta before start")
                print(value.get("text", ""), end="", flush=True, file=sys.stderr)
            elif event == "audio_chunk":
                if value.get("generation_id") != generation_id:
                    continue
                if not started or value.get("sequence") != sequence or value.get("sample_offset") != offset:
                    raise ValueError("discontinuous PCM stream")
                if (value.get("format"), value.get("sample_rate"), value.get("channels")) != ("pcm_s16le", 24000, 1):
                    raise ValueError("unsupported PCM format")
                pcm = base64.b64decode(value["data"], validate=True)
                if not pcm or len(pcm) % 2:
                    raise ValueError("invalid PCM samples")
                if not args.no_play:
                    if player is None:
                        player = subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "-loglevel", "error",
                                                   "-f", "s16le", "-ar", "24000", "-ch_layout", "mono", "-i", "pipe:0"],
                                                  stdin=subprocess.PIPE)
                    player.stdin.write(pcm)
                    player.stdin.flush()
                if output:
                    output.write(pcm)
                sequence += 1
                offset += len(pcm) // 2
            elif event == "error":
                raise ValueError(str(value.get("error", "generation failed")))
            elif event == "done":
                if value.get("generation_id") != generation_id:
                    continue
                audio = value.get("audio", {})
                if (not started or offset == 0 or value.get("finish_reason") != "stop"
                        or value.get("audio_finish_reason") != "stop"
                        or audio.get("samples") != offset or audio.get("chunks") != sequence):
                    raise ValueError("incomplete audio generation")
                complete = True
                break
        if not complete:
            raise ValueError("SSE closed without done")
        print(file=sys.stderr)
    finally:
        if not complete and player is not None:
            player.terminate()  # Flush buffered playback immediately on interruption.
        if response:
            response.close()
        if not complete:
            try:
                with post("/v1/omni/cancel", {"request_id": request_id}, timeout=5):
                    pass
            except Exception:
                pass
        if player is not None:
            try:
                player.stdin.close()
            except BrokenPipeError:
                pass
            if not complete:
                try:
                    player.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    player.kill()
            player.wait()
        if output:
            output.close()
            if not complete:
                args.save.unlink(missing_ok=True)
        if complete and player is not None and player.returncode != 0:
            raise RuntimeError("ffplay failed; check the selected audio output")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"\nVoice failed: {error}", file=sys.stderr)
        sys.exit(1)
