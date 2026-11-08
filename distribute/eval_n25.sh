#!/bin/sh
# N=25 rehearsal with the N=28 r8 parameters: PRESET=8, PARTS=4, CHUNKS=2.
# Two sections: r7 then r8; summarize after each.
# Expected N=25 total: 2207893435808352
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$DIR/.." && pwd)
cd "$ROOT"

rm -f distribute/n25_r7_part*.done distribute/n25_r8_part*.done

echo "================ N=25 r7 (PARTS=4 CHUNKS=2) ================"
for p in 0 1 2 3; do
    N=25 PRESET=7 PARTS=4 CHUNKS=2 sh distribute/run_part.sh "$p" 2 || exit 1
done
N=25 PRESET=7 PARTS=4 CHUNKS=2 sh distribute/summarize.sh
grep -h "CUDA kernel time" distribute/logs/n25_r7_part*_chunk*.log \
    | grep -oP 'CUDA kernel time: \K[0-9.]+' \
    | python3 -c "import sys;print('r7 kernel sum:', sum(map(float,sys.stdin.read().split()))/1000, 's')"

echo "================ N=25 r8 (PARTS=4 CHUNKS=2) ================"
for p in 0 1 2 3; do
    N=25 PRESET=8 PARTS=4 CHUNKS=2 sh distribute/run_part.sh "$p" 2 || exit 1
done
N=25 PRESET=8 PARTS=4 CHUNKS=2 sh distribute/summarize.sh
grep -h "CUDA kernel time" distribute/logs/n25_r8_part*_chunk*.log \
    | grep -oP 'CUDA kernel time: \K[0-9.]+' \
    | python3 -c "import sys;print('r8 kernel sum:', sum(map(float,sys.stdin.read().split()))/1000, 's')"
