#!/bin/sh
# 单卡正确性验证：对已知 N 跑 4 parts x CHUNKS 子块，与已知解数比较。
# 用法: sh distribute/validate_single.sh            # N=22, 2 chunks
#       N=20 sh distribute/validate_single.sh       # 最快
#       N=24 sh distribute/validate_single.sh       # 更强，单张 4090 约 2 小时
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
N=${N:-22}
PRESET=${PRESET:-7}
CHUNKS=${CHUNKS:-2}
PARTS=4
case "$N" in
  20) EXPECT=${EXPECT:-39029188884};;
  22) EXPECT=${EXPECT:-2691008701644};;
  24) EXPECT=${EXPECT:-227514171973736};;
  *)  EXPECT=${EXPECT:?set EXPECT for N=$N};;
esac
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "=== single-GPU validation: N=$N preset=$PRESET, $PARTS parts x $CHUNKS chunks (GPU=$CUDA_VISIBLE_DEVICES) ==="
rm -f "$DIR/n${N}_r${PRESET}_part"*"_chunk"*".done"
p=0
while [ "$p" -lt "$PARTS" ]; do
    N=$N PRESET=$PRESET CHUNKS=$CHUNKS PARTS=$PARTS sh "$DIR/run_part.sh" "$p" "$CHUNKS" || exit 1
    p=$((p+1))
done
python3 - "$DIR" "$N" "$PRESET" "$PARTS" "$CHUNKS" "$EXPECT" <<'PY'
import os, sys
d, n, r, parts, chunks, expect = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
total = 0
for p in range(parts):
    for c in range(chunks):
        total += int(open(os.path.join(d, f"n{n}_r{r}_part{p}_chunk{c}.done")).read())
print(f"chunk sum = {total}")
print(f"expected  = {expect}")
print("PASS" if total == expect else "FAIL")
sys.exit(0 if total == expect else 1)
PY
