#!/bin/sh
# build_qnx.sh — build mote for QNX 8.0, no make required.
#
#   Cross, on an x86_64 Linux or Windows host with QNX SDP 8.0 installed:
#     source ~/qnx800/qnxsdp-env.sh
#     scripts/build_qnx.sh x86_64        # the QEMU target image
#     scripts/build_qnx.sh aarch64le     # Raspberry Pi 5
#
#   Native, on the target (QNX Developer Desktop ships clang, not qcc):
#     scripts/build_qnx.sh native
#
# Output: build/qnx-<arch>/mote. The engine needs nothing beyond libc and libm;
# the thread pool in src/parallel.c is selected automatically on QNX, and
# src/platform.h turns on _QNX_SOURCE before any system header.
set -e
cd "$(dirname "$0")/.."

ARCH=${1:-x86_64}
CFLAGS=${CFLAGS:--O3 -std=c11 -Wall -Wextra}

case $ARCH in
    x86_64)
        CC="qcc -Vgcc_ntox86_64" ;;
    aarch64le)
        CC="qcc -Vgcc_ntoaarch64le"
        CFLAGS="$CFLAGS -march=armv8.2-a+dotprod" ;;   # Cortex-A76: SDOT
    native)
        CC=${CC:-clang}
        ARCH=$(uname -m) ;;
    *)
        echo "usage: $0 x86_64 | aarch64le | native" >&2; exit 2 ;;
esac

if [ "$1" != native ] && ! command -v qcc >/dev/null 2>&1; then
    echo "qcc not found: source qnxsdp-env.sh from your SDP 8.0 install first" >&2
    exit 1
fi

out=build/qnx-$ARCH
mkdir -p "$out"
# shellcheck disable=SC2086
$CC $CFLAGS -o "$out/mote" src/*.c -lm
echo "built $out/mote"
