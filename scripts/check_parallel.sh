#!/bin/sh
# check_parallel.sh — every parallel backend must print the same text.
#
#   scripts/check_parallel.sh [model] [tokenizer]
#
# Builds mote three ways into a scratch directory: serial (no threads), the
# POSIX worker pool (what QNX runs), and this host's default backend. Runs the
# same greedy prompt through each and fails unless the outputs are identical.
# Each matmul row is computed whole by one thread, so threading may change how
# fast the answer arrives and nothing else.
#
# Needs only a C compiler and a POSIX shell, so it runs on the target too (on
# the QNX Developer Desktop: CC=clang scripts/check_parallel.sh).
set -e
cd "$(dirname "$0")/.."

MODEL=${1:-models/stories15M.bin}
TOK=${2:-models/tokenizer.bin}
CC=${CC:-cc}
CFLAGS=${CFLAGS:--O3 -std=c11 -Wall -Wextra}
PROMPT=${PROMPT:-Once upon a time}
STEPS=${STEPS:-96}

[ -f "$MODEL" ] || { echo "no model at $MODEL (scripts/download_model.sh fetches one)"; exit 2; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

for b in serial pool default; do
    case $b in
        serial)  def=-DMOTE_SERIAL;   extra= ;;
        pool)    def=-DMOTE_PTHREADS; extra=-lpthread ;;
        default) def=;                extra= ;;
    esac
    # shellcheck disable=SC2086
    $CC $CFLAGS $def -o "$tmp/mote-$b" src/*.c -lm $extra
    "$tmp/mote-$b" "$MODEL" -z "$TOK" -t 0 -n "$STEPS" -i "$PROMPT" \
        > "$tmp/$b.txt" 2> "$tmp/$b.err"
    printf '%-8s %s\n' "$b" "$(grep 'tok/s' "$tmp/$b.err" | tr -d '[]')"
done

fail=0
for b in pool default; do
    if ! cmp -s "$tmp/serial.txt" "$tmp/$b.txt"; then
        echo "MISMATCH: $b differs from serial"
        diff "$tmp/serial.txt" "$tmp/$b.txt" | head -20
        fail=1
    fi
done
[ "$(wc -w < "$tmp/serial.txt")" -gt 10 ] || { echo "too little text came out"; fail=1; }
[ $fail -eq 0 ] && echo "ok: serial, pool and default agree ($(wc -w < "$tmp/serial.txt" | tr -d ' ') words)"
exit $fail
