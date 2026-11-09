#!/bin/sh
# Summarize the N=28 r8 campaign (PRESET=8, PARTS=4).
# Usage: sh distribute/sum28.sh [chunks]   (default 2)
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
CHUNKS=${1:-2}
N=28 PRESET=8 PARTS=4 CHUNKS=$CHUNKS sh "$DIR/summarize.sh"
