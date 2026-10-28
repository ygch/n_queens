# N=28 分布式求解：使用说明

本目录的脚本把一次求解切成 **4 个 part**（每组机器算一个），每个 part 内再切成若干
**chunk**，每个 chunk 结束后把它那份解数（精确 64 位整数）原子写入
`distribute/n<N>_r<preset>_part<P>_chunk<C>.done`。全部 chunk 的 `.done` 求和即为答案，
任何 chunk 中断后重跑都会跳过已完成的 `.done`，所以天然支持断点续算。

## 1. 原理

* **不预先生成子问题文件**。所有机器用同一份二进制、同一条命令，运行时由程序按
  `NQ_PARTS/NQ_PART/NQ_CHUNKS/NQ_CHUNK` 自行切分：先按 jkl 全序排序（排序含全序
  tie-breaker，保证跨机器一致），再取 `blk % PARTS == PART` 的 block 作为该 part，
  最后把该 part 的 block 列表按 chunk 连续切分。切分只依赖 `(N, preset, PARTS, PART,
  CHUNKS, CHUNK)`，与机器、时间无关，因此任意机器算出的 chunk 结果可互换、可重复。
* **part 之间无重叠、无遗漏**（模划分是完备划分），所以 4 个 part 的和就是全解数。
* **chunk 粒度同时影响容错和速度**。GPU 每卡只能驻留约 384 个 block；chunk 太小时
  每个 job 只有 1–2 个“波次”，总时间由该 job 里最重的一个 block 决定，重块的尾巴
  无法被其他工作摊薄。N=22 单卡实测（同一个 part 的全部 block）：
  1 份 18.1s、2 份 19.6s、4 份 22.3s、8 份 28.7s、**16 份 38.6s（2.13×）**。
  因此大 N 不要用 16 份；每个 job 至少覆盖 4/16 左右（每卡 ≥5 个波次）。
* `.done` 都是精确整数（不是浮点、不要用 `awk` 求和），最终用 `summarize.sh` 的
  Python 64 位求和。任意互不重叠的区间划分（`chunk<C>` 或 `chunk<C>-<D>`）都可以
  直接相加。

## 2. 编译

```bash
cd src && bash compile.sh
```

所有参与机器必须使用**同一份已提交的源码**（同一 git commit）。不同 commit 的 jkl
排序或计数逻辑若有差异，chunk 结果将不可合并。

## 3. 单卡正确性验证（务必先做）

在任意一张 4090/5090 上对已知解数的小 N 跑满 4 个 part × 2 个 chunk，并与 OEIS
精确值比较：

```bash
N=20 sh distribute/validate_single.sh     # 约 20 秒，期望 39029188884
N=22 sh distribute/validate_single.sh     # 约 1 分钟，期望 2691008701644
N=24 sh distribute/validate_single.sh     # 约 2 小时，期望 227514171973736
```

脚本会先删除该 N/preset 下的旧 `.done`，再依次调用 `run_part.sh 0..3`，最后打印
`PASS`/`FAIL`。`CUDA_VISIBLE_DEVICES` 默认 0，可用环境变量覆盖。

## 4. 正式运行（每组机器一个 part）

最简单：等当前 chunk 跑完、停掉旧 runner、更新代码并重新编译后，每组机器执行

```bash
nohup sh distribute/run_fast.sh $P > distribute/logs/part${P}_rest.out 2>&1 &
```

`run_fast.sh` 会自动识别 `distribute/` 下已完成的连续 chunk 前缀（例如 chunk0），
把**剩余全部区间作为一个大 job** 跑完；全新的 part（没有任何 `.done`）则跑整个 part。
不需要记 from/span/to 参数。

* 如果担心单次故障损失，可在 part 号后加 `2`（剩余对半跑，总时间约 +8%）：
  `sh distribute/run_fast.sh $P 2`。
* 程序内部会通过 `solve_multi_gpu()` 把该 part 的 block 按 `blk % gpu_num` 分给 8 张卡。
* 中途重启：重跑同一条命令；已完成前缀会被跳过，未完成的区间沿用同一个 `.done` 名。
* 若机器整组失效：把该 part 交给任意一组空闲机器重跑即可（`.done` 随任务保留）。
* 注意：`run_fast.sh` 检测到已有 `n_queens` 进程时会拒绝启动，避免与正在跑的 chunk 重叠。

### 4.1 手动指定区间（可选）

`run_span.sh <part> <chunks> <from> <span> [to]` 可以手动控制区间，例如
`sh distribute/run_span.sh 0 16 1 4 16` 依次跑 chunk 1-4、5-8、9-12、13-15。
一般不需要，`run_fast.sh` 已覆盖。

后台运行建议：

```bash
nohup sh distribute/run_fast.sh 0 > distribute/logs/part0_rest.out 2>&1 &
```

每台机器只跑一个 part，避免两张卡同时写同一个 `.done`。

## 5. 汇总

把所有组机器上的 `.done` 收集到同一台机器的 `distribute/` 下（文件名不冲突），然后：

```bash
sh distribute/summarize.sh
```

默认所有 part 都是 16 网格（`chunk0` + `chunk1-15` / `chunk0-15` 都能识别）。脚本会
检查每个 part 的区间无重叠、无遗漏（覆盖 `[0,16)`），打印每个 part 的部分和和
`TOTAL: <精确解数>`；有缺失/重叠时退出码为 1。只有确实混用了不同网格时才需要
`GRIDS=16,16,16,2` 之类的参数。

## 6. 监控（只读）

```bash
# 每 10 分钟打印：进程、8 卡利用率、当前 job 起始时间/已运行时间、已完成 .done
# BUDGET_H 是单个 job 的预计小时数，仅用于粗 ETA
N=28 BUDGET_H=190 sh distribute/watch.sh 600
```

`watch.sh` 不修改任何文件，不影响正在跑的任务。日志现在逐行 flush，启动几秒后就能
在 `distribute/logs/*.log` 里看到 `Constellation generation` / `work split` /
`checkpoint buffers` 三行；运行期间没有更多输出，用 `watch.sh` 看存活和进度。

## 7. 注意

* `distribute/*.done` 和 `distribute/logs/` 已被 `.gitignore` 忽略，不会误提交。
* 每个 chunk 的写入是 `写 .tmp -> rename` 的原子操作，中途崩溃不会留下半截数字；
  `run_part.sh`/`run_span.sh` 把“文件非空且非 0”作为完成判据，请勿手工创建/编辑
  `.done`。
* **不要用 16 个 chunk**：实测会让同一 part 慢约 2 倍（见第 1 节）。2–4 个大 job 的
  速度接近不切 chunk，同时保留数天级的故障粒度。
* 同一个 part 内不要混用不同的 `CHUNKS` 网格（`.done` 文件名里不含网格大小）；
  跨 part 可以不同，汇总时用 `GRIDS` 声明。
* 先跑 `validate_single.sh` 再上大机器；N=28 一旦开跑就是数天到数周量级。
