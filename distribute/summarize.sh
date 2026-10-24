#!/bin/sh
# 汇总 distribute/ 下一个 N/preset 的全部 .done（精确 64 位整数；不要用 awk）。
# 用法: N=28 PRESET=7 CHUNKS=16 sh distribute/summarize.sh
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
N=${N:-28}
PRESET=${PRESET:-7}
PARTS=${PARTS:-4}
CHUNKS=${CHUNKS:-16}
python3 - "$DIR" "$N" "$PRESET" "$PARTS" "$CHUNKS" <<'PY'
import os, sys
d, n, r, parts, chunks = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
total = 0; found = 0; missing = []
for p in range(parts):
    for c in range(chunks):
        f = os.path.join(d, f"n{n}_r{r}_part{p}_chunk{c}.done")
        if not os.path.exists(f):
            missing.append(f); continue
        txt = open(f).read().strip()
        if not txt:
            missing.append(f); continue
        total += int(txt); found += 1
print(f"N={n} preset={r}: found {found}/{parts*chunks} chunk files")
if missing:
    print(f"MISSING {len(missing)}:")
    for m in missing[:20]: print("  ", m)
    if len(missing) > 20: print("   ...")
print(f"TOTAL: {total}")
sys.exit(0 if not missing else 1)
PY
