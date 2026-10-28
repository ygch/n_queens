#!/bin/sh
# 运行一个 part 的一段连续 chunk 区间（可断点续跑；用于把若干小 chunk 合并成
# 较大的 job，摊薄“最重 block”的尾巴）。
#
# 用法: sh distribute/run_span.sh <part> <chunks> <from> <span> [to]
#   from  起始 chunk（含）
#   span  每个 job 覆盖的 chunk 数
#   to    结束 chunk（不含），默认 = chunks
#
# 例:
#   sh distribute/run_span.sh 0 16 1 5      # 依次跑 [1,6) [6,11) [11,16)
#   sh distribute/run_span.sh 0 16 13 3 16  # 只跑 [13,16)（最轻，先出结果）
#
# 每个 job 的输出: n< N >_r<PRESET>_part<P>_chunk<C>-<D>.done
# （span=1 时与旧命名 chunk<C>.done 一致）
# 环境变量: N(28) PRESET(7) PARTS(4) BIN
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$DIR/.." && pwd)
BIN=${BIN:-$ROOT/src/n_queens}
N=${N:-28}
PRESET=${PRESET:-7}
PARTS=${PARTS:-4}

PART=${1:-}; CHUNKS=${2:-}; FROM=${3:-}; SPAN=${4:-}; TO=${5:-}
[ -n "$PART" ] && [ -n "$CHUNKS" ] && [ -n "$FROM" ] && [ -n "$SPAN" ] || {
    echo "usage: $0 <part> <chunks> <from> <span> [to]" >&2; exit 2; }
[ -n "$TO" ] || TO=$CHUNKS

for v in "$PART" "$CHUNKS" "$FROM" "$SPAN" "$TO"; do
    case "$v" in ''|*[!0-9]*) echo "arguments must be non-negative integers" >&2; exit 2;; esac
done
[ "$PART" -lt "$PARTS" ] || { echo "part must be in [0,$((PARTS-1))]" >&2; exit 2; }
[ "$CHUNKS" -ge 1 ] || { echo "chunks must be >= 1" >&2; exit 2; }
[ "$SPAN" -ge 1 ] || { echo "span must be >= 1" >&2; exit 2; }
[ "$FROM" -lt "$TO" ] || { echo "need from < to" >&2; exit 2; }
[ "$TO" -le "$CHUNKS" ] || { echo "to must be <= chunks" >&2; exit 2; }
[ -x "$BIN" ] || { echo "binary not found/executable: $BIN" >&2; exit 2; }
grep -aq "span-split v1" "$BIN" || {
    echo "binary lacks span support; rebuild from the latest commit" >&2; exit 2; }

LOG=$DIR/logs
mkdir -p "$LOG"

# 旧的 0 值 .done 是失败运行的残留，直接删除
for f in "$DIR"/n${N}_r${PRESET}_part${PART}_chunk*.done; do
    [ -e "$f" ] || continue
    if [ ! -s "$f" ] || [ "$(cat "$f")" = "0" ]; then
        echo "remove invalid .done: $f"
        rm -f "$f"
    fi
done

# 检查 [a,b) 是否与已有 .done 区间冲突：完全覆盖则 SKIP(10)，部分重叠则报错(2)
check_range() {
    python3 - "$DIR" "$N" "$PRESET" "$PART" "$1" "$2" <<'PY'
import os, sys, re
d, n, r, p, a, b = sys.argv[1:7]
a, b = int(a), int(b)
pat = re.compile(r'^n%s_r%s_part%s_chunk(\d+)(?:-(\d+))?\.done$' % (n, r, p))
for f in sorted(os.listdir(d)):
    m = pat.match(f)
    if not m:
        continue
    x = int(m.group(1)); y = int(m.group(2)) if m.group(2) else x
    y += 1
    if x < b and a < y:
        if x == a and y == b:
            print("SKIP"); sys.exit(10)
        print("planned [%d,%d) overlaps existing %s [%d,%d)" % (a, b, f, x, y), file=sys.stderr)
        sys.exit(2)
sys.exit(0)
PY
}

echo "=== N=$N preset=$PRESET part=$PART/$PARTS chunks=$CHUNKS range=[$FROM,$TO) span=$SPAN ==="
c=$FROM
while [ "$c" -lt "$TO" ]; do
    e=$((c + SPAN))
    [ "$e" -gt "$TO" ] && e=$TO
    if [ "$e" -eq $((c + 1)) ]; then
        name=$c
    else
        name=$c-$((e - 1))
    fi
    out=$DIR/n${N}_r${PRESET}_part${PART}_chunk${name}.done
    log=$LOG/n${N}_r${PRESET}_part${PART}_chunk${name}.log

    if [ -s "$out" ]; then
        echo "[$(date '+%F %T')] skip chunk $name (done: $(cat "$out"))"
        c=$e
        continue
    fi

    check_range "$c" "$e"
    rc=$?
    if [ "$rc" -eq 10 ]; then
        echo "[$(date '+%F %T')] skip chunk $name (already covered)"
        c=$e
        continue
    fi
    if [ "$rc" -ne 0 ]; then
        echo "[$(date '+%F %T')] range [$c,$e) conflicts with an existing .done; abort" >&2
        exit 1
    fi

    echo "[$(date '+%F %T')] part $PART chunk $name start -> $log"
    NQ_PARTS=$PARTS NQ_PART=$PART NQ_CHUNKS=$CHUNKS NQ_CHUNK=$c NQ_CHUNK_SPAN=$((e - c)) \
        NQ_CHUNK_OUT="$out" "$BIN" "$N" "$PRESET" > "$log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ] || [ ! -s "$out" ]; then
        echo "[$(date '+%F %T')] part $PART chunk $name FAILED rc=$rc; see $log" >&2
        exit 1
    fi
    val=$(cat "$out")
    if [ "$val" = "0" ]; then
        echo "[$(date '+%F %T')] part $PART chunk $name FAILED: zero partial sum; see $log" >&2
        rm -f "$out"
        exit 1
    fi
    echo "[$(date '+%F %T')] part $PART chunk $name done: $val"
    c=$e
done
echo "range [$FROM,$TO) complete."
