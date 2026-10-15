# Omni extension release notes

## v0.1.1

- Standalone native-only documentation and historical validation.
- Non-loopback startup requires an API key; HTTP regression covers voice and cancel authentication.
- Prefer the known mixed-Q4 Talker file automatically; disclose the unresolved BF16 CUDA stall.
- Protocol-2 PCM example client with ordering checks, immediate deltas and Ctrl-C cancellation; voice regression updated to the current protocol.
- Per-file reference-kernel licensing/provenance and preserved upstream authorship.
- Full-source and upgrade-kit archives use regenerated source manifests and matching module contents.

Native phrase grouping, incremental GPU waveform decoding, cache growth, paging and generation IDs were already present in the preceding implementation. This release does not claim new speech-quality qualification.
