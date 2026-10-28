#!/bin/sh
# 汇总 distribute/ 下一个 N/preset 的全部 .done（精确整数；不要用 awk）。
# 支持任意互不重叠的 chunk 区间：chunk<C>.done 或 chunk<C>-<D>.done。
#
# 用法:
#   N=28 PRESET=7 GRIDS=16,16,16,2 sh distribute/summarize.sh
#   （GRIDS 给出每个 part 的 chunk 网格；缺省时所有 part 都用 CHUNKS，默认 16）
#   例如 4090 的三个 part 用 16 网格（chunk0 + span），5090 的 part3 用 2 网格。
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
N=${N:-28}
PRESET=${PRESET:-7}
PARTS=${PARTS:-4}
CHUNKS=${CHUNKS:-16}
GRIDS=${GRIDS:-}
python3 - "$DIR" "$N" "$PRESET" "$PARTS" "$CHUNKS" "$GRIDS" <<'PY'
import os, re, sys

d, n, r, parts, chunks, grids_arg = sys.argv[1:7]
parts, chunks = int(parts), int(chunks)
grids = [int(x) for x in grids_arg.split(',')] if grids_arg else [chunks] * parts
if len(grids) != parts:
    print(f"GRIDS must have {parts} comma-separated entries", file=sys.stderr)
    sys.exit(2)

pat = re.compile(r'^n%s_r%s_part(\d+)_chunk(\d+)(?:-(\d+))?\.done$' % (n, r))
files = {p: [] for p in range(parts)}
for f in sorted(os.listdir(d)):
    m = pat.match(f)
    if not m:
        continue
    p = int(m.group(1))
    x = int(m.group(2))
    y = int(m.group(3)) if m.group(3) else x
    if 0 <= p < parts:
        files[p].append((x, y + 1, f))

total = 0
ok = True
for p in range(parts):
    g = grids[p]
    ranges = sorted(files[p])
    cover = 0
    part_sum = 0
    problems = []
    for x, y, f in ranges:
        if x < cover:
            problems.append(f"overlap [{x},{y}) {f}")
        elif x > cover:
            problems.append(f"gap [{cover},{x}) before {f}")
        cover = max(cover, y)
        try:
            part_sum += int(open(os.path.join(d, f)).read().strip())
        except Exception:
            problems.append(f"bad value in {f}")
    if cover != g:
        problems.append(f"coverage ends at {cover}, grid {g}")
    status = "  OK" if not problems else "  <-- " + "; ".join(problems)
    print(f"part{p}: {len(ranges)} file(s), grid={g}, sum={part_sum}{status}")
    ok = ok and not problems
    total += part_sum

print(f"TOTAL: {total}")
print("OK" if ok else "INCOMPLETE/OVERLAP")
sys.exit(0 if ok else 1)
PY
