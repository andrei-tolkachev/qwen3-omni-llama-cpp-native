# Omni extension release notes

## v0.2.0 — True Full Duplex (2026-10-16)

- Independent Thinker and GPU speech workers with lossless PCM/WAV fragment joining, image input, bounded queues and revision fences.
- Adaptive append-only heard history, phrase playback acknowledgements, cancellation after synthesis and recomputation of input dependent on unheard output.
- Auxiliary Thinker queries, final-input replacement, protected context compare-and-swap and exact-text speech continuations.
- Speech-first memory placement, allocation-driven paging, bounded projection workspaces, shorter phrases, synthesis pacing and exact per-generation Talker KV-prefix reuse.
- Optional quoted-text history view for a single inference, rebuilt on revision changes without rewriting the stored conversation.
- Keep turn-local context immediately before the current input without mutating committed history or adding retrieval text to speech conditioning.
- Remove obsolete 3787/4096-row conditioning guards; preserve the loaded context limits and multimodal rows.
- Native concurrency, capture, HTTP authentication and numerical regressions; matching full-source/upgrade-kit archives and updated English/Russian documentation.
- Native API scope only: capture, playback and automatic conversation policy are caller responsibilities. Historical evidence and fresh release checks are separated.


## v0.1.1

- Standalone native-only documentation and historical validation.
- Non-loopback startup requires an API key; HTTP regression covers voice and cancel authentication.
- Prefer the known mixed-Q4 Talker file automatically; disclose the unresolved BF16 CUDA stall.
- Protocol-2 PCM example client with ordering checks, immediate deltas and Ctrl-C cancellation; voice regression updated to the current protocol.
- Per-file reference-kernel licensing/provenance and preserved upstream authorship.
- Full-source and upgrade-kit archives use regenerated source manifests and matching module contents.

Native phrase grouping, incremental GPU waveform decoding, cache growth, paging and generation IDs were already present in the preceding implementation. This release does not claim new speech-quality qualification.
