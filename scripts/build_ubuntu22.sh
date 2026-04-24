#!/usr/bin/env bash
# Build the mlsys binary inside an Ubuntu 22.04 container so the resulting
# ELF is compatible with the contest harness (Ubuntu 22.04 LTS, glibc 2.35).
# libstdc++ and libgcc are statically linked so the binary has no extra
# C++ runtime dependency; glibc stays dynamic as intended.
#
# Output: build-ubuntu22/mlsys
#
# Requires: podman (or docker). Run from the repo root.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if command -v podman >/dev/null 2>&1; then
    RUNTIME=podman
elif command -v docker >/dev/null 2>&1; then
    RUNTIME=docker
else
    echo "error: neither podman nor docker is installed" >&2
    exit 1
fi

echo ">>> Building mlsys inside ubuntu:22.04 using $RUNTIME"

"$RUNTIME" run --rm \
    -v "$REPO_ROOT":/work:Z \
    -w /work \
    ubuntu:22.04 \
    bash -c '
        set -e
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq
        apt-get install -y --no-install-recommends \
            build-essential cmake ca-certificates \
            nlohmann-json3-dev >/dev/null

        rm -rf build-ubuntu22
        cmake -S . -B build-ubuntu22 \
            -DCMAKE_BUILD_TYPE=Release \
            -DSTATIC_STDLIB=ON
        cmake --build build-ubuntu22 -j2 --target mlsys

        echo
        echo ">>> ldd build-ubuntu22/mlsys:"
        ldd build-ubuntu22/mlsys || true
    '

echo
echo "Binary ready at: $REPO_ROOT/build-ubuntu22/mlsys"
