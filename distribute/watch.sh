#!/bin/sh
# 只读监控：进程、8 卡利用率、当前 job 已运行时间、已完成 .done 与部分和。
# 不修改任何文件，不影响正在跑的任务。
#
# 用法: sh distribute/watch.sh [interval_sec] [part]
#   interval_sec 默认 600 秒；part 可选（只看该 part）
# 环境变量: N(28) PRESET(7) BUDGET_H（单个 job 的预计小时数，用于粗 ETA，0=不显示）
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
N=${N:-28}
PRESET=${PRESET:-7}
BUDGET_H=${BUDGET_H:-0}
INTERVAL=${1:-600}
ONLY=${2:-}
case "$INTERVAL" in ''|*[!0-9]*) echo "interval must be an integer" >&2; exit 2;; esac

while :; do
    printf '===== %s =====\n' "$(date '+%F %T')"

    if pgrep -x n_queens >/dev/null 2>&1; then
        ps -C n_queens -o pid=,args= | sed 's/^/  proc /'
    else
        echo "  proc: no n_queens process"
    fi
    if pgrep -af 'run_span\.sh|run_part\.sh' >/dev/null 2>&1; then
        echo "  runner: alive"
    else
        echo "  runner: not found"
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw \
            --format=csv,noheader 2>/dev/null | sed 's/^/  gpu /'
    fi

    python3 - "$DIR" "$N" "$PRESET" "$ONLY" "$BUDGET_H" <<'PY'
import os, re, sys, time
d, n, r, only, budget_h = sys.argv[1:6]
budget_h = float(budget_h)
only = int(only) if only else None

pat = re.compile(r'^n%s_r%s_part(\d+)_chunk(\d+)(?:-(\d+))?\.done$' % (n, r))
parts = {}
for f in sorted(os.listdir(d)):
    m = pat.match(f)
    if not m:
        continue
    p = int(m.group(1))
    if only is not None and p != only:
        continue
    x = int(m.group(2)); y = int(m.group(3)) if m.group(3) else x
    try:
        v = int(open(os.path.join(d, f)).read().strip())
    except Exception:
        v = 0
    parts.setdefault(p, []).append((x, y, f, v))

if parts:
    for p in sorted(parts):
        rs = sorted(parts[p])
        s = sum(v for _, _, _, v in rs)
        rng = ", ".join(f"[{x},{y}]" for x, y, _, _ in rs)
        print(f"  part{p}: {len(rs)} done, sum={s}, ranges {rng}")
else:
    print("  done: none yet")

logdir = os.path.join(d, 'logs')
if os.path.isdir(logdir):
    logs = []
    for f in os.listdir(logdir):
        m = re.match(r'^n%s_r%s_part(\d+)_chunk.*\.log$' % (n, r), f)
        if not m:
            continue
        if only is not None and int(m.group(1)) != only:
            continue
        p = os.path.join(logdir, f)
        logs.append((os.path.getmtime(p), p, f))
    if logs:
        mtime, path, name = max(logs)
        start = None
        blocks = None
        ktime = None
        with open(path, errors='ignore') as fh:
            for line in fh:
                if start is None:
                    m = re.search(r'\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?:\.\d+)?\]', line)
                    if m:
                        start = m.group(1)
                m = re.search(r'work split:.*-> (\d+) blocks, (\d+) constellations', line)
                if m:
                    blocks = int(m.group(1))
                m = re.search(r'CUDA kernel time: ([0-9.]+)ms', line)
                if m:
                    ktime = float(m.group(1))
        print(f"  current: {name}  blocks={blocks}  log_age={ (time.time()-mtime)/3600:.2f}h")
        if start:
            st = time.mktime(time.strptime(start, "%Y-%m-%d %H:%M:%S"))
            elapsed_h = (time.time() - st) / 3600.0
            print(f"  start: {start}  elapsed: {elapsed_h/24:.2f} days ({elapsed_h:.1f}h)")
            if ktime is not None:
                print(f"  last job finished in {ktime/3600000:.2f}h")
            elif budget_h > 0:
                pct = 100.0 * elapsed_h / budget_h
                eta = time.strftime("%m-%d %H:%M", time.localtime(st + budget_h * 3600))
                print(f"  budget: {budget_h:.0f}h -> {pct:.0f}%  rough ETA {eta} (local time)")
PY

    sleep "$INTERVAL"
done
