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
* **chunk 粒度只影响容错**：chunk 越小，单次失败损失越小，但启动/收尾开销略增。
  N=28 默认 16 个 chunk（每 part 约 1.4 天/chunk，单次故障最多损失约 1.4 天）。
* `.done` 都是精确整数（不是浮点、不要用 `awk` 求和），最终用 `summarize.sh` 的
  Python 64 位求和。

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

假设 4 组机器，每组 8 卡。第 P 组执行：

```bash
N=28 PRESET=7 PARTS=4 CHUNKS=16 sh distribute/run_part.sh $P
```

（`P` 取 0/1/2/3；默认值就是 N=28、PRESET=7、PARTS=4、CHUNKS=16，可直接
`sh distribute/run_part.sh 0`。）

* 程序内部会通过 `solve_multi_gpu()` 把该 part 的 block 按 `blk % gpu_num` 分给 8 张卡。
* 每完成一个 chunk，`distribute/n28_r7_part<P>_chunk<C>.done` 出现，日志在
  `distribute/logs/`。
* 中途重启：直接重跑同一条命令，已存在的 `.done` 会被跳过。
* 若机器整组失效：把该 part 交给任意一组空闲机器，用同样的 `PART` 值重跑即可，
  已完成的 chunk 不会重算（前提是 `.done` 文件随任务一起保留/拷贝）。

后台运行建议：

```bash
nohup sh distribute/run_part.sh 0 > distribute/logs/part0.out 2>&1 &
```

每台机器只跑一个 part，避免两张卡同时写同一个 `.done`。

## 5. 汇总

把所有组机器上的 `.done` 收集到同一台机器的 `distribute/` 下（文件名不冲突），然后：

```bash
N=28 PRESET=7 PARTS=4 CHUNKS=16 sh distribute/summarize.sh
```

输出 `found N/64 chunk files`、缺失列表（若有）和 `TOTAL: <精确解数>`。
只有 `found 64/64` 时 `TOTAL` 才是 N=28 的最终答案。

## 6. 注意

* `distribute/*.done` 和 `distribute/logs/` 已被 `.gitignore` 忽略，不会误提交。
* 每个 chunk 的写入是 `写 .tmp -> rename` 的原子操作，中途崩溃不会留下半截数字；
  但 `run_part.sh` 以“文件非空”为完成判据，请勿手工创建/编辑 `.done`。
* 不要把 CHUNKS 设得过大（例如 >64）：每个 chunk 都有固定的启动与收尾开销，
  N=28 默认 16 个；8 个也够用（故障损失上限约 2.8 天）。
* 先跑 `validate_single.sh` 再上大机器；N=28 一旦开跑就是数天到数周量级。
