#!/bin/sh
# 最简单入口：把某个 part 剩下没算的部分作为一个大 job 跑完。
#
# 用法: sh distribute/run_fast.sh <part> [jobs]
#   part  0..3（必需）
#   jobs  把剩余区间拆成几个 job，默认 1（不用管）
#
# 自动识别 distribute/ 下已经完成的连续 chunk 前缀（例如 chunk0），跳过它们；
# 其余区间一次跑完，输出一个 .done（如 chunk1-15.done）。
# 汇总: sh distribute/summarize.sh（默认 16 网格，无需额外参数）
# 环境变量: N(28) PRESET(7) PARTS(4) CHUNKS(16) BIN
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$DIR/.." && pwd)
BIN=${BIN:-$ROOT/src/n_queens}
N=${N:-28}
PRESET=${PRESET:-7}
PARTS=${PARTS:-4}
CHUNKS=${CHUNKS:-16}

PART=${1:-}
JOBS=${2:-1}
[ -n "$PART" ] || { echo "usage: $0 <part> [jobs]" >&2; exit 2; }
case "$PART" in ''|*[!0-9]*) echo "part must be an integer" >&2; exit 2;; esac
case "$JOBS" in ''|*[!0-9]*) echo "jobs must be an integer" >&2; exit 2;; esac
[ "$PART" -lt "$PARTS" ] || { echo "part must be in [0,$((PARTS-1))]" >&2; exit 2; }
[ "$JOBS" -ge 1 ] || { echo "jobs must be >= 1" >&2; exit 2; }
[ -x "$BIN" ] || { echo "binary not found/executable: $BIN" >&2; exit 2; }

if pgrep -x n_queens >/dev/null 2>&1; then
    echo "another n_queens job is running; wait until it finishes" >&2
    exit 1
fi

# 0 值 .done 是失败运行的残留
for f in "$DIR"/n${N}_r${PRESET}_part${PART}_chunk*.done; do
    [ -e "$f" ] || continue
    if [ ! -s "$f" ] || [ "$(cat "$f")" = "0" ]; then
        echo "remove invalid .done: $f"
        rm -f "$f"
    fi
done

# 已完成的连续前缀 [0,K)
K=$(python3 - "$DIR" "$N" "$PRESET" "$PART" "$CHUNKS" <<'PY'
import os, re, sys
d, n, r, p, chunks = sys.argv[1:6]
p, chunks = int(p), int(chunks)
pat = re.compile(r'^n%s_r%s_part%s_chunk(\d+)(?:-(\d+))?\.done$' % (n, r, p))
rs = []
for f in sorted(os.listdir(d)):
    m = pat.match(f)
    if not m:
        continue
    x = int(m.group(1)); y = int(m.group(2)) if m.group(2) else x
    rs.append((x, y + 1, f))
rs.sort()
cover = 0
for x, y, f in rs:
    if x != cover:
        print(f"existing {f} is not a contiguous prefix (expected start {cover}); "
              f"use run_span.sh or delete it", file=sys.stderr)
        sys.exit(2)
    cover = y
if cover > chunks:
    print(f"prefix ends at {cover}, beyond grid {chunks}", file=sys.stderr)
    sys.exit(2)
print(cover)
PY
) || exit 1

if [ "$K" -ge "$CHUNKS" ]; then
    echo "part $PART already complete ($K/$CHUNKS chunks)"
    exit 0
fi

REM=$((CHUNKS - K))
SPAN=$(( (REM + JOBS - 1) / JOBS ))
echo "part $PART: $K/$CHUNKS chunks done; running [$K,$CHUNKS) as $JOBS job(s), span=$SPAN"
exec sh "$DIR/run_span.sh" "$PART" "$CHUNKS" "$K" "$SPAN" "$CHUNKS"
