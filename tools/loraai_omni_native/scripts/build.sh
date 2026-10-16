#!/usr/bin/env bash
set -euo pipefail
omni_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
llama_source=${1:?Usage: build.sh LLAMA_SOURCE [LLAMA_BUILD]}
llama_build=${2:-$llama_source/build}
cmake -S "$omni_source" -B "$omni_source/build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_SOURCE_DIR="$llama_source" -DLLAMA_BUILD_DIR="$llama_build"
cmake --build "$omni_source/build" --parallel "${OMNI_BUILD_JOBS:-4}"
cmake --build "$omni_source/build" --target omni-duplex-test omni-voice-capture-test --parallel "${OMNI_BUILD_JOBS:-4}"
