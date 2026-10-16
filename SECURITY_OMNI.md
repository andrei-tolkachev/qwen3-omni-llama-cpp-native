# Security

`omni-server` refuses non-loopback binding without a nonempty `--api-key` or `--api-key-file`. `localhost` is pinned to a literal loopback address. With a configured key, upstream HTTP middleware protects `/v1/omni/voice`, `/v1/omni/cancel` and `/v1/omni/duplex` plus all its control routes. `/health` remains public. `omni-server-security-test` exercises missing, wrong and valid keys for all ten voice/cancel/duplex routes using that middleware.

An API key is shared service access, not per-user isolation. Use TLS, request-size limits and suitable authorization/rate limits for remote deployments. Do not expose an unauthenticated loopback service through a port-forward or reverse proxy. Store keys outside the checkout; prefer `--api-key-file` over process arguments. Avoid verbose request logging with private input.

Cancellation acknowledges the ID match; it does not assert that a running GPU kernel has already ended. Wait for the voice response to terminate before a new generation. Clients must clear buffered playback and ignore stale generation IDs immediately.

Camera/screen frames are supplied by the caller. Pointer annotations and screen-request actions never authorize OS capture or input.

Do not include credentials, model weights, private audio, prompts or device captures in issue reports. Use private vulnerability reporting when available. This experimental extension has not received a comprehensive independent security assessment.

A duplex session owns one model lease. Disconnect or `/duplex/close` cancels and joins both inference workers before releasing it. Keep generation/revision fences and derive playback ACKs from actual output, not received bytes. Context replacement protects the active dialogue tail; source text and images remain caller-provided data.
