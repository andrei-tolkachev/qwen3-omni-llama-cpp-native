# Native Qwen3-Omni speech extension

Independent C++/GGML support for **Qwen3-Omni-30B-A3B-Instruct** speech output on top of llama.cpp: Talker, Code Predictor and Code2Wav, mixed-Q4 Talker, streamed text and PCM, cancellation and GPU memory management. Audio/image input uses upstream mtmd/Thinker. Offline conversion is Python; inference is native C++.

**llama.cpp, its server and GGML belong to their upstream authors.** This module contains the Omni additions by Andrei Tolkachev under [MIT](LICENSE). AI assisted the implementation. It is not an official llama.cpp or Qwen release.

Qualified base: `86632248188c106d749fad34a1dcd237c95863d4`. The server adapter currently requires Linux ELF interposition. GPU measurements cover Linux x86-64/CUDA/RTX 5090 only; device selection uses available memory, not a GPU product name. Other backends need qualification.

## Build

Install into a clean checkout (the full-source archive already contains this installation):

```sh
python3 scripts/apply_to_llama.py /path/to/llama.cpp --check
python3 scripts/apply_to_llama.py /path/to/llama.cpp
cmake -S /path/to/llama.cpp -B /path/to/llama.cpp/build-omni \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_OMNI=ON -DGGML_CUDA=ON
cmake --build /path/to/llama.cpp/build-omni --target omni-server omni-generate omni-decode omni-quantize -j4
```

Choose CUDA architectures appropriate to the destination machine with the normal CMake options; no architecture is pinned by the extension. Executables from an integrated build are in `build-omni/bin`. A matching prebuilt shared-library llama build can also be linked with a separate CMake invocation using `-DLLAMA_SOURCE_DIR=/path/to/llama.cpp` and `-DLLAMA_BUILD_DIR=/path/to/llama.cpp/build-omni`. The integrated build above is the recommended, better-tested path. Never mix binaries/libraries from different builds.

## Weights

Use a Qwen3 Omni Thinker GGUF plus the matching multimodal projector. Convert the three speech components from the original BF16 Hugging Face model directory:

```sh
python3 scripts/convert_speech.py --hf /models/Qwen3-Omni-30B-A3B-Instruct \
  --out /models/omni/omni-experimental --dry-run
python3 scripts/convert_speech.py --hf /models/Qwen3-Omni-30B-A3B-Instruct \
  --out /models/omni/omni-experimental
```

The converter is standard-library Python and packs the 128 Talker experts without numerical conversion. It checks the qualified HF index/config hashes and all 733 tensor shapes/byte ranges before writing. It never overwrites existing outputs. Historical `staging` names/metadata are retained for compatibility with the independently checked CPU reference. These component GGUF files are supported by this extension, not by an unmodified llama-server. Model weights are excluded from the archives; their original licenses continue to apply.

### Recommended mixed-Q4 Talker

Build `omni-quantize` and convert the packed BF16 Talker into a separate file:

```sh
build-omni/bin/omni-quantize \
  /models/omni/omni-experimental/talker-experts-packed-bf16-staging.gguf \
  /models/omni/omni-experimental/talker-experts-packed-Q4.gguf
python3 scripts/check_quantized_talker.py \
  /models/omni/omni-experimental/talker-experts-packed-bf16-staging.gguf \
  /models/omni/omni-experimental/talker-experts-packed-Q4.gguf
export OMNI_TALKER_MODEL=/models/omni/omni-experimental/talker-experts-packed-Q4.gguf
```

The mixed Q4 profile quantizes 40 gate/up expert matrices to Q4_K and 20 down matrices to Q4_0. Down projections have 384 input columns, which cannot form complete 256-element Q4_K blocks. All 357 other tensors (attention, router, shared experts, speech projections, embeddings, head and normalization) stay byte-identical BF16. This is **mixed Q4 Talker, not every speech component in Q4**. Predictor and Code2Wav remain BF16; Code2Wav expands to F32 on GPU for reference accuracy.

Talker GGUF shrinks from 6,649,234,624 to 2,308,130,080 bytes (6.19 to 2.15 GiB). The GPU uses the quantized weights directly, including selected-expert paging; it does not expand all experts back to BF16/F32. Sensitive one-dimensional weights still use F32. The converter refuses existing destinations, requantization and malformed input. It expands only one tensor at a time during offline conversion and needs no GPU. Original weights remain available for rollback.

When `OMNI_TALKER_MODEL` is unset, the module prefers `talker-experts-packed-Q4.gguf` if present. Otherwise it uses the original BF16 file and prints a warning. An explicit override always wins; a damaged Q4 file fails instead of silently selecting another model. The memory policy measures the selected file.

**BF16 limitation:** one captured input stalled in CUDA expert multiplication. Q4 completed that same input in resident and staged modes; this does not establish a fix for the BF16 kernel. The mixed profile has no importance matrix and still needs listening and language qualification.

## Run and stream

```sh
/path/to/llama.cpp/build-omni/bin/omni-server \
  -m /models/omni/thinker.gguf --mmproj /models/omni/mmproj.gguf \
  --host 127.0.0.1 --port 8081 -c 65536 -ngl auto --fit on --fit-target 1024
```

Use a smaller context if the available memory requires it. **Non-loopback binding is refused without `--api-key` or `--api-key-file`.** With a key configured, both `/v1/omni/voice` and `/v1/omni/cancel` require it. Prefer a protected key file and TLS for remote access. `/health` remains public. A shared API key is not per-user authorization.

A working protocol-2 request (no key is needed for the loopback command above):

```sh
curl --no-buffer --fail-with-body http://127.0.0.1:8081/v1/omni/voice \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"0123456789abcdef0123456789abcdef","generation_id":"1123456789abcdef0123456789abcdef","stream":true,"audio_protocol":2,"messages":[{"role":"user","content":"Say hello briefly."}]}'
```

For live playback, install `ffplay` and run from the module directory:

```sh
python3 scripts/voice_client.py --text 'Say hello briefly.'
# Optional: add --wav input.wav, --api-key-file /path/to/key, or --save reply.pcm.
# Without an audio device: --no-play --save reply.pcm
# Replay saved PCM:
ffplay -f s16le -ar 24000 -ch_layout mono -autoexit reply.pcm
```

The client prints each text delta immediately, validates PCM ordering, writes directly to the player/file, and cancels the native request on Ctrl-C. `OMNI_API_KEY` is also accepted by the client. Failed or cancelled saved output is removed. Use unique IDs for every new generation.

```text
event: start
data: {"request_id":"...","generation_id":"..."}

event: delta
data: {"text":"Hel"}

event: delta
data: {"text":"lo"}

event: phase
data: {"phase":"speech"}

event: audio_chunk
data: {"generation_id":"...","sequence":0,"sample_offset":0,"format":"pcm_s16le","sample_rate":24000,"channels":1,"data":"BASE64_PCM"}

event: done
data: {"generation_id":"...","text":"Hello","finish_reason":"stop","audio":{"format":"pcm_s16le","sample_rate":24000,"channels":1,"samples":12345,"chunks":1},"audio_finish_reason":"stop","speech_backend":"gpu","memory_mode":"resident"}
```

`done` above is abbreviated. Protocol 2 requires `stream:true` and returns audio totals, **not a second full WAV**. Events `start`, `audio_chunk` and `done` carry `generation_id`; text deltas belong to their enclosing response stream. `error` is terminal; a disconnected stream without `done` is incomplete. Protocol 1/non-stream JSON returns a complete WAV in `audio.data` for compatibility and accumulates it in host RAM. Use protocol 2 for long replies.

Cancel with POST `/v1/omni/cancel`, body `{"request_id":"..."}`, using the same authentication. Disconnect also cancels. Thinker checks cancellation while generating; speech checks between GPU graphs, layers and frames. A currently running kernel finishes before it can stop. The cancel response is an acknowledgement, not a GPU-completion barrier: wait for the active voice response to terminate before starting another turn. The server serializes inference. Cancellation of a non-active request ID does not cancel a different request. Clients must clear playback immediately and reject stale generations; the server cannot flush a remote sound device.

Input audio is a recorded turn using upstream `input_audio` blocks. This API is not a continuous full-duplex microphone transport. Speaker IDs 2301–2303 select native voices. Optional `speech_seed` (0–4294967295) makes codec sampling repeatable. Incomplete text, exhausted speech frames and missing EOS report errors instead of a successful partial reply.

## Long replies and memory

Phrase grouping is implemented **inside `src/voice.cpp` in this module**. It aims for complete phrases near 160 text tokens, with a word-boundary fallback around 256 tokens. Each phrase keeps the original input conditioning and speaker. These are synthesis work batches, not reply-length limits; unusual text without word boundaries can produce larger batches. All phrases share one generation ID and continuous PCM sequence. The phrase join overlaps 10 ms without inserting silence; within-phrase samples are unchanged.

Code2Wav decodes at most 16 codec frames per push (about 1.28 seconds of audio), retains causal decoder state and releases unused workspaces. Protocol 2 does not retain the full waveform. Talker KV starts at 512 positions and grows while preserving prior keys/values, bounded by its 65,536-position window. Total text/prompt state is still finite.

A segmented stress run produced **600.97 seconds of PCM** in 212.97 seconds. A separate **12-minute unsegmented run ended with unintelligible speech despite valid PCM and EOS**. The long duration is a transport/memory result, not proof of speech quality. Only selected short Russian regions were checked with ASR; broad listening, English and other hardware remain unqualified. Raw mathematical formulas degraded speech in one long test.

| Setting | Behavior |
| --- | --- |
| `OMNI_BACKEND` | `gpu` by default; `cpu` explicitly selects reference math; no automatic CPU speech fallback |
| `OMNI_DEVICE` | GGML device name; otherwise the GPU with most available memory |
| `OMNI_MEMORY_MODE` | `auto`, `resident`, or `staged`; auto may unload Thinker under memory pressure after complete text generation and tokenization |
| `OMNI_SPEECH_MODELS` | Component directory; default `omni-experimental` beside Thinker |
| `OMNI_TALKER_MODEL` | Explicit Talker path; otherwise prefer the known mixed-Q4 filename, then BF16 |
| `OMNI_GPU_RESERVE_MB` | Default 512 MiB headroom |
| `OMNI_GPU_BUDGET_MB` | Optional admission budget minus resident weights, also checked before KV/decoder growth; **not** a hard total-VRAM cap |

Weights are paged by layer when needed; Talker loads selected experts. Speech runs on one chosen GPU, without speech tensor parallelism. Predictor and decoder have separate lifetimes. Very small devices can fail when the minimum graph/cache working set does not fit. Thinker unload is a fallback after text completion; partial Thinker state is not discarded. Reload occurs on the next request.

## Context and visual input

A 65,536-token context was checked using a synthetic 64,200-token prompt. The server receives exactly the messages supplied by the caller; it does not add persistent memory or automatic conversation compaction. Latest-user hidden states already attend to the earlier prompt. Talker capture is bounded to 3,787 rows and fails explicitly when exceeded. 128K is not qualified.

`image_url` blocks can accompany the current audio/text turn. No camera or screen capture runs inside the server. Optional `observe_only:true` runs Thinker without speech and supports the same cancellation route. This is sampled-image observation, not a temporal video decoder.

For annotations, pass a trusted `visual_frame_id` and ask for spoken text followed by `<lora_pointer>JSON</lora_pointer>`, with JSON `null` or `{x,y,label}` in normalized coordinates. The adapter constrains and strips that suffix from speech/text and emits a `pointer` event bound to the frame ID, also returned in `done.pointers`. Clients must discard annotations after source changes or cancellation. A pointer is an annotation, not an OS input command.

With no current screen frame, `allow_screen_request:true` permits the suffix `<lora_pointer>{"request_screen":true}</lora_pointer>` and returns `actions:[{"type":"request_screen"}]`. This requests permission from the client; it does not grant capture or device-control access.

## Measurements and qualification

Historical results from [VALIDATION.json](VALIDATION.json), on Linux/CUDA/RTX 5090, are not a new benchmark of this release.

| Test | Mode / warm state | First text delta | Complete result |
| --- | --- | --- | --- |
| Four short Russian Q4 phrases | GPU resident; per-run warm state not recorded | Not recorded | 2.26–5.41 s |
| Two BF16 sentence controls | GPU staged; per-run warm state not recorded | Not recorded | 9.95–10.91 s |
| Synthetic 60-token input | Q4 GPU resident; already loaded | 0.032 s | 1.954 s |
| Synthetic 64,200-token input | Q4 GPU resident; already loaded, 65,536 context | 12.510 s | 14.438 s |
| Segmented long stream, 2,469 text tokens | GPU resident; warm state not recorded | First PCM: 26.217 s | 212.974 s for 600.967 s PCM |
| Native cancel during text / speech | Original native tests | — | 0.009 / 0.101 s |

The unsegmented 12-minute quality failure above applies when interpreting the long-stream row. Timing numbers measured through separate application stacks have deliberately not been mixed into this native-only table.

GPU numerical reference: two recurrent frames match exactly; 12,885 decoder samples have max absolute error 7.97e-7 against retained independent fixtures. Chunked decoding tests retained sample counts with >59 dB SNR. Q4 resident/paged codes match each other for two frames; Q4 is not claimed numerically equivalent to BF16. ASR checks of four short Russian phrases are not a listening benchmark.

## Upgrades and checks

Apply the standalone kit to a fresh upstream checkout with `scripts/apply_to_llama.py --check` first. A changed patch context stops for review; differing module files need explicit `--replace`. Rebuild the whole matching binary/library set.

Only `src/server_adapter.cpp` couples to server internals: it captures Thinker `l_out-23` / hidden_states[24] via eval callback and Linux ELF `llama_decode` interposition. Graph geometry and APIs must be checked on upgrades. The CUDA patch honors `GGML_PREC_F32` and disables TF32 for these cuBLAS operations; re-evaluate it when upstream changes precision handling.

```sh
python3 scripts/verify_source_manifest.py
python3 scripts/check_upgrade.py --models /models/omni --build /path/to/build-omni/bin
OMNI_GPU_BUDGET_MB=4096 python3 scripts/check_upgrade.py --models /models/omni --build /path/to/build-omni/bin
python3 tests/check_codec_stream.py --model /models/omni/omni-experimental/code2wav-staging-bf16.gguf --decoder /path/to/build-omni/bin/omni-decode
python3 tests/check_voice_dialogue.py --url http://127.0.0.1:8081 --wav /path/to/synthetic-test.wav
```

`check_upgrade.py` uses BF16 fixtures; it is a numerical probe, not broad BF16 stability qualification. Build/run `omni-cache-growth-test` with the Predictor GGUF for KV growth comparison; `omni-audio-join-test` and `omni-visual-reply-test` need no weights. `omni-server-security-test` uses the real upstream HTTP middleware on an ephemeral loopback port, without model weights, to check authentication on voice and cancel routes.
