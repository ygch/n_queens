#!/bin/sh
# 运行一个 part 的全部子块（可断点续跑）。
# 用法: sh distribute/run_part.sh <part 0..3> [chunks]
# 默认: N=28, PRESET=7, PARTS=4, CHUNKS=16; .done 写到 distribute/。
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$DIR/.." && pwd)
BIN=${BIN:-$ROOT/src/n_queens}
N=${N:-28}
PRESET=${PRESET:-7}
PARTS=${PARTS:-4}
CHUNKS=${CHUNKS:-16}
PART=${1:-}
[ -n "$PART" ] || { echo "usage: $0 <part 0..$((PARTS-1))> [chunks]" >&2; exit 2; }
[ -n "${2:-}" ] && CHUNKS=$2
case "$PART" in ''|*[!0-9]*) echo "part must be an integer" >&2; exit 2;; esac
[ "$PART" -lt "$PARTS" ] || { echo "part must be in [0,$((PARTS-1))]" >&2; exit 2; }
case "$CHUNKS" in ''|*[!0-9]*) echo "chunks must be an integer" >&2; exit 2;; esac
[ "$CHUNKS" -ge 1 ] || { echo "chunks must be >=1" >&2; exit 2; }
[ -x "$BIN" ] || { echo "binary not found/executable: $BIN" >&2; exit 2; }
LOG=$DIR/logs
mkdir -p "$LOG"
echo "=== N=$N preset=$PRESET part=$PART/$PARTS chunks=$CHUNKS bin=$BIN ==="
c=0
while [ "$c" -lt "$CHUNKS" ]; do
    out=$DIR/n${N}_r${PRESET}_part${PART}_chunk${c}.done
    log=$LOG/n${N}_r${PRESET}_part${PART}_chunk${c}.log
    if [ -s "$out" ]; then
        echo "[$(date '+%F %T')] skip chunk $c (done: $(cat "$out"))"
        c=$((c+1)); continue
    fi
    echo "[$(date '+%F %T')] part $PART chunk $c start -> $log"
    NQ_PARTS=$PARTS NQ_PART=$PART NQ_CHUNKS=$CHUNKS NQ_CHUNK=$c \
        NQ_CHUNK_OUT="$out" "$BIN" "$N" "$PRESET" > "$log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ] || [ ! -s "$out" ]; then
        echo "[$(date '+%F %T')] part $PART chunk $c FAILED rc=$rc; see $log" >&2
        exit 1
    fi
    echo "[$(date '+%F %T')] part $PART chunk $c done: $(cat "$out")"
    c=$((c+1))
done
echo "part $PART complete."
