# True Full Duplex: native session API (protocol 1)

`omni-server` can process new input with Thinker while native Talker/Predictor/Code2Wav continue producing speech on GPU. Enable this explicitly with `OMNI_DUPLEX=1`. The ordinary `/v1/omni/voice` API remains available between sessions. Incoming audio and images can be processed while speech synthesis continues. This package provides native inference/session primitives; microphone capture, playback and conversation policy belong to the caller.

## Start

```sh
OMNI_DUPLEX=1 OMNI_MEMORY_MODE=resident build-omni/bin/omni-server \
  -m /models/thinker.gguf --mmproj /models/mmproj.gguf \
  --host 127.0.0.1 --port 8081 -c 65536 -ngl auto --fit on
```

The GPU speech weights are placed **before** Thinker fitting. Llama therefore sees their real allocation when placing Thinker weights/KV; at least 2048 MiB is left as fit headroom for the simultaneous working sets. Existing larger `--fit-target` values are preserved. The physical Thinker microbatch is capped at 128 to limit prefill contention. There is one Thinker context and one speech worker, each with independent backend streams, KV and mutable graph state. Speech weights remain cached across responses. Under pressure, the manager frees unused graph workspaces first and evicts unused weights only when a measured KV/page/decoder allocation does not fit; it does not reload all weights at a fixed free-memory threshold. No model quantization, speech sampling algorithm or codec arithmetic is changed by duplex scheduling. Thinker uses the configured sampling temperature unless the session overrides it.

`auto` and `resident` memory modes keep Thinker loaded for the entire session. `staged` is rejected. Idle model sleep and context shifting are disabled in duplex mode. `-ngl auto --fit on` can place some Thinker layers in host memory when necessary; speech remains on GPU. Existing layer/selected-expert paging remains available through `OMNI_GPU_BUDGET_MB`. This is a resident-weight accounting budget, not a total VRAM cap. Context is finite: no history or model precision is silently discarded to make a request fit. A minimum working-set failure is explicit. Latency depends on hardware, input size and paging; the measured 5090 results do not guarantee equal speed on smaller GPUs.

At least four HTTP workers are reserved for the event stream and control/input requests. One session owns the model until its stream closes. Conflicting legacy requests or a second session receive HTTP 409, rather than consuming all HTTP workers waiting for the model.

Speech conditioning is bounded by the loaded Thinker context and Talker's 65,536-position capacity. The former 3,787/4,096-row capture limits could reject a valid user turn containing retrieved evidence, audio or video even with VRAM available. Projection workspaces now process at most 1,024 rows per batch; all rows and multimodal states are retained. Talker reserves positions for its prefix and generated codec frames and reports actual context exhaustion explicitly. See `CONDITIONING_VALIDATION.json` for regression and memory measurements. Long conditioning still increases Talker prefill time.

## Open and send input

POST `/v1/omni/duplex` with a JSON object. The response is a long-lived SSE stream:

```json
{"messages":[],"adaptive_dialogue":true,"temperature":0.8,"max_tokens":512,"speaker_id":2301,"speech_seed":42}
```

`messages` is optional initial history. `max_tokens` defaults to -1 (the model context/EOS limit); `max_audio_frames` defaults to 65536 per phrase. Both text and speech exhaustion remain explicit errors. `speech_seed` is optional. The `session` event supplies a server-issued `session_id`, `next_sequence:0`, `revision:0`, and `playback_ack_required:true`. Callers should check the advertised capabilities `query`, `replace_input`, `visual`, `context`, `interrupt_input`, `playback_final`, and `input_context` before enabling the corresponding features.

POST `/v1/omni/duplex/input` from another connection:

```json
{"session_id":"ID_FROM_SESSION_EVENT","sequence":0,"final":false,"content":[{"type":"text","text":"Remember seven. "}]}
```

```json
{"session_id":"ID_FROM_SESSION_EVENT","sequence":1,"final":true,"content":[{"type":"text","text":"What is that number plus five?"}]}
```

Sequences are session-wide, contiguous, zero-based. Invalid requests do not consume a sequence. HTTP 409 means a sequence conflict; HTTP 429 means a finalized input is still waiting to become the next spoken response. Retry that same sequence after `speech_started` for the pending turn. At most one pending finalized turn and one currently speaking response are retained. An empty `content:[]` is allowed only to finalize a nonempty open turn.

For microphone input, send native PCM fragments:

```json
{"session_id":"ID","sequence":2,"final":false,"content":[{"type":"input_audio","input_audio":{"format":"pcm16","sample_rate":16000,"channels":1,"data":"BASE64_S16LE"}}]}
```

Alternatively each fragment can be a complete PCM S16LE `format:"wav"` container, using the ordinary upstream `input_audio` shape. Rates from 8000 through 96000 Hz and one/two channels are accepted. The format must remain constant across adjacent audio fragments. Adjacent audio fragments are concatenated **without resampling, dropping or modifying samples**, then passed to mtmd as one WAV. Adjacent text blocks are concatenated byte-for-byte. `image_url` blocks are also passed through to mtmd. Compressed audio fragments are not accepted by this API.

Thinker processes the latest cumulative input snapshot; superseded partial results are fenced and queued fragments are coalesced instead of building an inference backlog. Partial inference starts at most ten times per second and produces a preview of at most 32 tokens (96 in adaptive mode). Finalized input bypasses this coalescing interval. Sending 100-500 ms microphone fragments avoids needless frontend work. Final inference uses the whole input and full captured conditioning, with no partial-answer text inserted into history. The existing upstream audio frontend re-encodes cumulative audio snapshots; this is not a new stateful streaming audio encoder. `final` is supplied by the client/VAD. Echo cancellation, microphone permissions and acoustic barge-in detection belong to the client.

An input with `replace:true` atomically replaces the accumulated fragments with the supplied authoritative content. A client can submit the complete VAD WAV plus captured camera/screen frames for the final input, avoiding duplicated microphone pre-roll. Optional `history_text` is the transcript retained in subsequent dialogue history; the current final inference still receives the original audio and images. Optional `max_tokens` reserves the measured remaining context for that turn. `visual:{"frame_id":"...","allow_screen_request":true}` binds screen pointers/actions to a captured frame. Technical visual markup is removed from spoken text and returned as `pointers`/`actions` on the generation events.

## Events and playback

- `input_started`: accepted input sequence/revision begins inference.
- `input_processed`: sequence, revision, preview/final text, `final`, `thinker_ms`, and a generation ID for a final result. This result is provisional until `speech_started`.
- `speech_started`: generation ID, input sequence, dialogue revision and the immutable text spoken by this generation.
- `audio_chunk`: generation ID, contiguous sequence and sample offset, mono PCM S16LE, 24000 Hz, base64 data. Samples append to one player; no final WAV is replayed.
- `phrase_boundary`: generation ID, `text_end` (UTF-8 byte offset) and `sample_end`. The phrase becomes eligible for history acknowledgement after those samples have actually played. A boundary may arrive before the final 10 ms tail is emitted; use playback sample counts, not arrival of this event.
- `speech_done`: synthesis finished, total samples/chunks and `speech_ms`. **It does not mean the user has heard all the audio.**
- `keepalive`: idle stream heartbeat.
- `error`: terminal session failure. Open a new session after closing the old stream; do not replay failed audio as a successful response.

POST `/v1/omni/duplex/ack` periodically from the player:

```json
{"session_id":"ID","generation_id":"GEN","played_samples":24000,"final":false}
```

Offsets must be monotonic and cannot exceed emitted samples. Send `final:false` until `speech_done` has been received; then send `final:true` when acknowledging playback. The optional flag defaults to true for compatibility with earlier benchmark clients. Releasing the speech lane requires synthesis completion, all samples acknowledged and a final ACK. This prevents a last interim ACK from releasing the next generation before the caller processes `speech_done`. Thinker continues handling new input while playback is pending. A client that only stores/benchmarks audio can acknowledge consumed samples to simulate playback; an interactive client must acknowledge the actual playback position. A stalled player therefore cannot cause unlimited speech to accumulate.

## Barge-in and coherent history

First stop/flush the local player and acknowledge its last played offset; then POST `/v1/omni/duplex/cancel`:

```json
{"session_id":"ID","generation_id":"GEN"}
```

The response immediately fences that generation and returns a new dialogue `revision`. Cancel works during synthesis **and after `speech_done` while playback is pending**. The native worker finishes an executing GPU graph before releasing its own state. A duplicate/stale cancel cannot stop a newer generation. Reject late PCM for the cancelled generation and late provisional text whose revision precedes the returned fence.

History retains the original user turn and only complete assistant phrases covered by acknowledged playback. An unacknowledged or partially played phrase is conservatively excluded; word-level forced alignment is not claimed. The pending input remains intact. Any Thinker result conditioned on the unheard continuation is discarded and recomputed at the new revision. This also covers a final response already queued behind the interrupted speech.

The cancel receipt includes `heard_text`, `heard_samples` and `played_samples` for durable history/audio persistence. To interrupt an input before speech promotion, use `/v1/omni/duplex/interrupt-input` with its latest `sequence` and `content`. Empty content discards an unfinalized VAD misfire; nonempty content preserves the interrupted user question. Sequence -1 is allowed before any fragment has been submitted. If promotion raced with interruption, the matching generation is cancelled; later generations are fenced.

## Auxiliary queries and context maintenance

Optional input `context` is an ephemeral UTF-8 string for retrieved memory, web evidence and previous visual observations. Native inserts it as a separate system message immediately before the current user input for that input's Thinker inference only. Earlier messages remain byte-for-byte unchanged; turn-local evidence stays close to the input that it qualifies. It is included in the input byte limit, excluded from user speech conditioning, and never committed as a dialogue message. Callers must treat external materials as data, measure the full input including this context, and keep original audio, current frames and the user's question in `content`. No speech-model math or model precision changes are required. Sending thousands of reference tokens as user speech instead forces Talker to prefill them serially before the first PCM chunk. See `CONTEXT_VALIDATION.json` for the measured regression and fix.

Optional final-input `history_mode:"quoted_text"` (capability `quoted_history`) changes only the current inference view. Leading system messages remain messages; the exact remaining user/assistant records are serialized into a `dialogue_history` JSON object in a temporary system message. Current `context` and the original current multimodal input follow it. The authoritative conversation, playback acknowledgements and persisted heard text are unchanged. The view is rebuilt from the latest revision on every recomputation, so it cannot retain a cancelled draft. Default `"messages"` preserves the normal conversation template; the next input defaults back to that mode.

Quoted mode requires text-only prior user/assistant content (strings or text blocks); non-text history is rejected rather than silently discarded. Supply `history_text` when recording audio/image turns that may later use this mode. Current audio/images are still processed normally. Callers must measure this exact view, including JSON overhead, with `/query` and `operation:"context_tokens"`; the server does not silently truncate it. Choosing when to use this view belongs to the caller. No conversation policy is embedded in the transformation.

POST `/v1/omni/duplex/query` with `{"session_id":"ID","query":{"messages":[...],"max_tokens":768}}`. ASR, web planning and visual observation use the same Thinker worker, with one bounded pending query; Talker continues independently. Disconnecting the query cancels its work. Queries do not append dialogue messages. `operation:"context_tokens"` inside `query` uses exact upstream multimodal tokenization and returns `context_tokens` (the capacity) alongside measured prompt tokens.

POST `/v1/omni/duplex/context` with `{"session_id":"ID"}` to read `messages`, `revision` and `preserve_tail`. To replace history, send `messages` and the exact observed `revision` to the same route. A changed revision or altered protected tail returns 409. Keep at least the last four messages and all messages from the active user/assistant pair onward, as specified by `preserve_tail`. Replacement invalidates provisional Thinker results but leaves the active Talker generation untouched.

Close with POST `/v1/omni/duplex/close`, body `{"session_id":"ID"}`, or disconnect the SSE stream. Both workers are cancelled and joined before another session is admitted. All routes use the server's existing API-key authentication.

Limits: 8 MiB per HTTP/input turn, 64 content blocks per request/accumulated turn after joining adjacent audio/text, 32 MiB serialized history, 64 output events and 4 MiB queued event data. History/context exhaustion is explicit; callers can implement compaction through the context API. These limits bound memory without setting a total session-duration limit.

## Upgrade checks

Build `omni-duplex-test` and `omni-server-security-test` in addition to the native server/tools. They cover real parallel workers, revision rollback, final playback fences, lossless PCM joining/replacement, auxiliary queries, context compare-and-swap, early input interruption, visual metadata, invalid offsets/sequences, bounded queues, disconnect, worker failure and authentication without model weights. Run the numerical fixtures and existing voice dialogue regression too.

Against an isolated real-model server started with `OMNI_DUPLEX=1`:

```sh
python3 tools/loraai_omni_native/tests/check_duplex.py \
  --url http://127.0.0.1:8081 --output duplex-results.json --wav /path/to/test.wav
```

This checks continuing PCM after new input has been processed, serial/concurrent seeded PCM identity, latency, cancellation/recomputation, and optional fragmented audio. Test clients acknowledge consumed PCM instead of using a sound device. `DUPLEX_VALIDATION.json` records the earlier native scheduling qualification; `ADAPTIVE_DIALOGUE_VALIDATION.json` records subsequent native adaptive-mode checks. Historical results are labelled by their scope; release preparation checks are in the root `RELEASE_VALIDATION.json`. Physical microphone/speaker echo cancellation and broad listening quality have not been qualified by these automated tests.

`omni-voice-capture-test` additionally covers conditioning beyond the old 4K limit, exact text/multimodal row retention, latest-user boundaries and real context limits.

### Observed CUDA qualification limit

One serial speech run during extended qualification aborted with CUDA Xid 11 on NVIDIA driver 595.99.02, before any overlapping input. The cause is unconfirmed and is not claimed fixed by this update. NVIDIA classifies the event in its [Xid catalog](https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html); the code alone does not establish the underlying cause. Subsequent native stress and instrumented results are recorded in `DUPLEX_VALIDATION.json`. Keep this distinction when qualifying another GPU/driver.

Compute Sanitizer completed serial and concurrent synthesis with the same PCM and no observed invalid-memory-access diagnostic. Its CUDA API diagnostics included recoverable status 910 from upstream's `cudaGraphExecUpdate` fallback, which explicitly clears that status and re-instantiates the graph. One initial instrumented concurrent run exceeded the test client's normal 180-second deadline; it completed after that diagnostic timeout was raised. Instrumented runtimes are not performance measurements.

## Adaptive mode and immutable played prefixes

Open with `"adaptive_dialogue":true` to opt in. The native session advertises `adaptive_dialogue`,
`committed_prefix` and `verbatim_speech`. Older callers keep their existing
behavior. Callers must check these capabilities before using the adaptive APIs.

In this mode the complete generated answer remains private to its speech job.
The assistant history entry starts empty. Playback ACKs append only completed
phrases covered by the actual playback offset; the entry never contains the
unheard suffix. Acknowledging another phrase advances the context revision and
invalidates provisional inference against an older prefix. Cancellation may
remove an empty placeholder, but cannot shorten or replace committed text.
`context` still protects the active tail against compaction. `record_input:false`
on `input` or `interrupt-input` allows an internal continuation to use current
input without inventing a new user message. Applications must use it only for
continuations tied to a real user request.

Adaptive synthesis uses smaller phrases (12-token target, 64-token maximum at
word boundaries) and limits generated PCM to six seconds ahead of playback plus
one codec chunk. Clients should retain at most half a second in the hardware player.
When dialogue policy requests a revision, the client finishes the next available
phrase boundary, cancels the remaining samples, and starts the revised response.
A boundary already passed to hardware cannot be selected for a cut. Manual
hang-up can stop mid-phrase; that incomplete phrase is conservatively omitted
because word-level forced alignment is not provided.

The Talker reuses only the exact identical projected conditioning prefix within
one generation. Its KV position rewinds to that prefix and overwrites the suffix;
Predictor/codec speech state is independent for each phrase. No cache is shared
between generations. The GPU cache regression compares hidden states exactly
against fresh prefill, including after KV growth. Model weights, precision and
sampling rules are unchanged, but smaller phrase segmentation can change prosody
and seeded PCM, so legacy/adaptive audio identity is not claimed.

Growing partial audio no longer invalidates every running preview. A completed
snapshot may publish its own sequence while newer fragments wait; authoritative
final input, cancellation and history revision changes still fence stale work.
Adaptive previews are ordinary provisional Thinker completions (up to 96 tokens);
no built-in conversational intent classifier is installed. This remains cumulative
re-encoding, not a streaming encoder with persistent audio frontend state.

### Exact speech and caller policy

Optional `spoken_text` on final input with `record_input:false` requests an exact
utterance, bounded to 1024 UTF-8 bytes without line breaks. A literal grammar and
text equality check run before Talker. Normal conditioning still comes from the
resident Thinker; no additional TTS model is loaded. This can support spoken tool
progress when the caller supplies real execution events and an appropriate text.
The native server does not execute tools or invent progress announcements.

Automatic decisions to continue, yield, revise or ask for clarification belong
to the application. VAD activity does not itself call cancellation in this server.
Use `/query` for auxiliary interpretation, playback ACKs for the heard prefix,
and `/cancel` plus a new input for a revised continuation. A new image can be
processed while speech continues, but a meaningful scene change does not by
itself interrupt the output. No application policy, UI, microphone driver or
acoustic echo cancellation is bundled.

See `ADAPTIVE_DIALOGUE_VALIDATION.json` for native measurements and limits.
