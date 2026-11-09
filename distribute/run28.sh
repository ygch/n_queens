#!/bin/sh
# N=28 r8 campaign: PRESET=8, PARTS=4, CHUNKS=2 (the N=25 rehearsal parameters).
# Usage: sh distribute/run28.sh <part 0..3> [chunks]
#   chunks defaults to 2; use 4 only if the 2-chunk checkpoint does not fit.
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
PART=${1:-}
CHUNKS=${2:-2}
[ -n "$PART" ] || { echo "usage: $0 <part 0..3> [chunks]" >&2; exit 2; }
[ "$CHUNKS" -ge 1 ] 2>/dev/null || { echo "chunks must be >= 1" >&2; exit 2; }

N=28 PRESET=8 PARTS=4 CHUNKS=$CHUNKS sh "$DIR/run_part.sh" "$PART" "$CHUNKS"
