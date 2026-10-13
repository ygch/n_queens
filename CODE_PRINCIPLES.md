# faf-cuda 分支代码原理详解

## 一、整体思路

本求解器采用 **D4 对称群 + 星座方法（Constellation Method）+ Jeff Somers 位并行回溯** 三层加速策略：

1. **D4 对称群缩减**：N-Queens 棋盘有 8 种对称操作（4 次旋转 x 2 次镜像）。只搜索对称等价类中的代表解，最终结果乘以对称倍数（8/4/2），将搜索空间缩小约 8 倍。

2. **星座方法**：在棋盘四条边上预先放置 4 个"边界皇后"——第一行皇后 i、最后一行皇后 j、第一列皇后 k、最后一列皇后 l。这 4 个皇后定义了一个"星座"（constellation），它们不仅减少了需要搜索的行数，还通过其对角线约束提前裁剪了大量无效分支。CPU 端再用回溯法放置若干"预设皇后"（preset_queens，默认 6 个），将棋盘状态（ld/rd/col）传给 GPU。

3. **GPU 位并行回溯**：每个 GPU 线程处理一个星座子问题，使用 Jeff Somers 的位并行回溯算法。用 3 个 32 位掩码 `ld`（左对角线）、`rd`（右对角线）、`col`（列占用）表示棋盘状态，一次位运算即可找到当前行所有可用位置。当 32 位移位导致对角线位溢出时，用 `ld_mem`/`rd_mem` 捕获丢失的位。

### 数据流

```
CPU: generate_constellations(n, preset_queens) → 生成所有星座子问题
 → padWithPseudo() → 按 jkl 分组排序、填充伪星座
 → 预计算 jkl_queens（边界皇后的对角线约束）
 → 传输到 GPU

GPU: n_queens_kernel<N>() → 每个线程处理一个星座，位并行回溯
 → 返回每个星座的解数

CPU: 汇总 results[i] * symmetry(ijkl) → 总解数
```

### 棋盘坐标系与位掩码约定

```
行号: 0 (顶) ~ N-1 (底)
列号: 0 (右) ~ N-1 (左)

位掩码: bit 0 = 最右列, bit N-1 = 最左列
  col  : 列占用, bit c = 1 表示列 c 已有皇后
  ld   : 左对角线 (\方向), 下一行整体左移1位 (<<1)
  rd   : 右对角线 (/方向), 下一行整体右移1位 (>>1)

边界皇后:
  i = 第一行(row 0)的皇后列号
  j = 最后一行(row N-1)的皇后列号
  k = 第一列(col 0)的皇后行号
  l = 最后一列(col N-1)的皇后行号
```

---

## 二、逐文件、逐函数解释

### 1. `main.cpp` — 命令行入口

```
用法: ./n_queens <N> [preset_queens]
  N: 棋盘大小，范围 [6, 28]
  preset_queens: CPU预设皇后数，默认6，范围 [4, N-2]
```

**`main(argc, argv)`**：解析命令行参数，验证范围，调用 `cuda_solve(N, preset_queens)`，打印结果和耗时。

---

### 2. `gpu_solver.h` — 求解器接口

仅声明 `long long cuda_solve(int n, int preset_queens)`，供 `main.cpp` 调用。

---

### 3. `utils.h` / `utils.cpp` — 工具函数

**`time_diff_ms(start, end)`**：计算两个 timeval 之间的毫秒差。

**`print_with_time(fmt, ...)`**：带时间戳的 `printf`，格式为 `[YYYY-MM-DD HH:MM:SS.mmm] message`。

---

### 4. `constellation.h` / `constellation.cpp` — D4 对称星座生成

#### 数据结构

**`Constellation`** 结构体：
- `ld, rd, col`：棋盘对角线/列的位掩码状态（CPU预设皇后放置后的状态）
- `start_ijkl`：高 12 位存 `start`（回溯起始行），低 20 位存 `ijkl` 编码
- `solutions`：解数（GPU写出）

#### ijkl 编码

**`ijkl`** = `(i << 15) | (j << 10) | (k << 5) | l`，20 位整数，编码 4 个边界皇后的位置：
- `i`：第一行的皇后列号
- `j`：最后一行的皇后列号
- `k`：第一列的皇后行号
- `l`：最后一列的皇后行号

每个值占 5 bit（0-31），支持 N ≤ 32。

#### 函数

**`toIjkl(i, j, k, l)`**：将 4 个边界皇后位置编码为 ijkl 整数。

**`geti/getj/getk/getl(ijkl)`**：从 ijkl 中解码出 i/j/k/l。

**`getJkl(ijkl)`**：返回 `ijkl & 0x7FFF`（低 15 位，即 jkl 三元组）。用于排序分组——相同 jkl 的星座共享同一组 jkl_queens，可复用共享内存。

**`rot90(n, ijkl)`**：将 ijkl 顺时针旋转 90 度。变换关系：`i' = N-1-k, j' = N-1-l, k' = j, l' = i`。

**`mirvert(n, ijkl)`**：左右镜像。变换关系：`i' = N-1-i, j' = N-1-j, k' = l, l' = k`。

**`jAsMin(n, ijkl)`**：通过旋转和镜像，将星座规范化为"j 最小"的表示。具体做法是找到 4 个边界皇后中离角最近的那个，通过旋转将其放到最后一行（j 位置），再镜像使 j 靠右。这是 D4 对称消除的第一步——确保每个等价类只有一个代表。

**`checkRotations(n, ijklList, i, j, k, l)`**：检查 `(i,j,k,l)` 的 90°/180°/270° 旋转是否已存在于集合中。若已存在则跳过，避免重复枚举。

**`symmetry90(n, ijkl)`**：判断星座是否具有 90 度旋转对称性。

**`symmetry(n, ijkl)`**：计算对称倍数。
- 如果 `i + j == N-1` 且 `k + l == N-1`（180度对称）：
  - 如果还有 90 度对称 → 返回 2（只有 4 个等价解，而非 8 个）
  - 否则 → 返回 4
- 否则 → 返回 8（一般情况，8 个等价解）

最终汇总时 `sum += results[i] * symmetry(n, ijkl)`。

**`getLD(ijkl)`**：计算边界皇后 j 和 l 产生的左下对角线掩码。`g_L >> j` 是 j 的左对角线，`g_L >> l` 是 l 的左对角线。

**`getRD(ijkl)`**：计算边界皇后 j 和 k 产生的右下对角线掩码。`g_L >> j` 是 j 的右对角线，`1 << k` 是 k 的右对角线。

**`placePresetQueens(ijkl, ld, rd, col, row, queens)`**：递归回溯，在内部行（非边界行）放置 preset_queens 个预设皇后。
- 跳过边界行（`row == k` 或 `row == l`）——这些行已有边界皇后。
- 当放置够 preset_queens 个皇后时，将当前状态 `(ld, rd, col)` 和起始行 `row` 存入 `g_subConstellations`。
- 每行计算可用位置 `free = ~(ld | rd | col | jkl对角线约束) & mask`，逐个尝试（位扫描）。

**`generateIjkls(ijklList)`**：枚举所有不重复的边界皇后配置 (i,j,k,l)。
- **无角皇后情况**：4 层循环枚举 j, l, k, i，跳过角和冲突位置，用 `checkRotations` 去重。
- **有角皇后情况**：固定角 (0,0)，枚举 k 和 i。通过 D4 对称，一个角皇后的情况等价于所有 4 个角。

**`generate_constellations(n, preset_queens)`**：主入口函数。
1. 设置全局变量（g_n, g_L, g_mask, g_presetQueens）
2. 调用 `generateIjkls` 生成所有不重复的边界配置
3. 对每个 ijkl：
   - `jAsMin` 规范化
   - 计算 4 个边界皇后的初始 ld/rd/col
   - 调用 `placePresetQueens` 放置预设皇后，生成子问题
4. 返回所有子问题列表

---

### 5. `n_queens_cuda.cu` — CUDA 求解器（核心）

#### 数据结构

**`Constellation4`**：GPU 端的星座数据，比 CPU 端 `Constellation` 更紧凑：
```cpp
struct Constellation4 {
    int ld, rd, col, start_ijkl;  // 16 bytes, 无 solutions 字段
};
```

#### 常量

**`WG_SIZE = 64`**：每个线程块的线程数（2 个 warp）。

#### GPU 内核：`n_queens_kernel<N>()`

**模板参数 N**：棋盘大小。模板化使得编译器可以将 N 作为常量优化（如 `1 << (N-1)` 变成编译时常量），并为每个 N 生成最优代码。

**`__launch_bounds__(WG_SIZE, 24)`**：告诉编译器每块 64 线程，目标每 SM 24 块。实际受共享内存限制为 9 块/SM（18 warp）。

**输入**：
- `constellations[]`：每个线程一个 Constellation4
- `jkl_queens_arr[]`：每个 workgroup 的 jkl 对角线约束数组（N 个 uint）
- `results[]`：输出，每个线程一个解数

**内核逻辑**：

1. **加载数据**：`gid`（全局线程索引），`lid`（块内线程索引）。加载星座数据到寄存器：`ld`, `rd`, `col`, `start_ijkl`。`col` 的初始化 `~(L-2) ^ c.col`：`~(L-2)` 设置边界位（bit 0 和 bit N-1 为 1），`^ c.col` 合入 CPU 端的列占用。

2. **加载 jkl_queens 到共享内存**：`__shared__ unsigned int jkl_queens[N]`——每个块共享一份。前 N 个线程协作加载，`__syncthreads` 等待完成。

3. **处理伪星座**：`start == 69` 是伪星座标记（填充用），直接返回 0。

4. **修正边界皇后的对角线**：皇后 k 在第一列，其对角线从左上方延伸，需根据起始行 `start` 修正 `ld`。皇后 l 在最后一列，其对角线从右上方延伸，需根据起始行 `start` 修正 `rd`。

5. **初始化回溯**：`free_val = ~(ld | rd | col | jkl_queens[row])`——计算起始行的可用位置。`queen = (-free_val) & free_val`——取最低可用位（位扫描：二进制补码技巧）。`queens[lid][start] = queen`——存入共享内存栈。

6. **主回溯循环** — Jeff Somers 位并行 DFS。每次迭代有三种情况：

   **a) 叶子节点**（`row == N-2` 且 `free_val != 0`）：`solutions++`，回溯一行。

   **b) 前进**（`free_val != 0` 且非叶子）：选最低可用位，压栈，进入下一行。更新对角线：`ld_mem = __funnelshift_l(ld, ld_mem, 1)` 捕获 ld 的第 31 位，`rd_mem = __funnelshift_r(rd_mem, rd, 1)` 捕获 rd 的第 0 位，`ld = (ld | queen) << 1`，`rd = (rd | queen) >> 1`。

   **c) 回溯**（`free_val == 0`）：退回上一行，从栈中弹出 queen，恢复 ld/rd/ld_mem/rd_mem。

   **统一尾代码**：
   - `free_val = ~(jkl_queens[row] | ld | rd | col)`：计算当前行的可用位置
   - `free_val &= ~(queen + direction - 1)`：防重复访问掩码。前进(dir=1)给 `~queen`，回溯(dir=0)给 `~(queen-1)`
   - `col ^= queen`：XOR 翻转——前进时加入列占用，回溯时移除

7. **写回结果**：`results[gid] = solutions`

#### 关键位运算技巧

- **位扫描** `queen = (-free) & free`：`-free` 的补码会保留最低位的 1，AND 后得到最低可用位。
- **`__funnelshift_l(lo, hi, s)`** = `(hi << s) | (lo >> (32-s))`：32 位移位时捕获溢出位。第一个参数是低位，第二个是高位（易混淆）。映射到 PTX `shf.l.wrap.b32` 指令。
- **`__funnelshift_r(lo, hi, s)`** = `(lo >> s) | (hi << (32-s))`：反向移位。
- **`col ^= queen`**：XOR 翻转技巧。前进时 `col` 加入皇后的列，回溯时同一个 XOR 操作移除该列。避免了 if/else 分支。
- **ld_mem/rd_mem**：当 `ld << 1` 时，bit 31 会丢失。`ld_mem` 用 funnelshift 捕获这个位，在回溯时用 `__funnelshift_r` 恢复。这使得 32 位掩码可以表示超过 32 行的对角线状态。

#### Host 函数

**`sortByJkl(consts)`**：按 `getJkl(ijkl)` 排序星座。相同 jkl 的星座被分到同一个 workgroup，共享同一份 `jkl_queens` 共享内存，减少全局内存访问。

**`padWithPseudo(consts, n, wgSize)`**：
1. 先按 jkl 排序
2. 在不同 jkl 组之间插入伪星座，使每组的 constellation 数量为 `wgSize` 的倍数
3. 伪星座的 `start_ijkl = (69 << 20)`，内核检测到 `start == 69` 直接返回 0
4. 确保每个 workgroup 内所有线程处理相同 jkl 的星座

**`LAUNCH_N(n_val)` 宏**：为每个 N 值生成一个 `case` 分支：
1. `cudaFuncSetAttribute(..., cudaSharedmemCarveoutMaxShared)`：请求最大共享内存
2. 以 `<<<numWg, WG_SIZE>>>` 启动模板化的内核

**`cuda_solve(n, preset_queens)`**：主入口函数：
1. 调用 `generate_constellations` 生成所有子问题
2. `padWithPseudo` 排序填充
3. 预计算 `jkl_queens_arr`：对每个 workgroup，根据其首个星座的 ijkl 计算边界皇后 j/k/l 在每一行的对角线约束
4. 将 `Constellation` 转换为紧凑的 `Constellation4`
5. 分配 GPU 内存，传输数据
6. `switch(n)` 通过 `LAUNCH_N` 宏启动对应 N 的模板内核
7. 同步等待完成，拷回结果
8. 汇总：`sum += results[i] * symmetry(n, ijkl)`，跳过伪星座

### 6. `compile.sh` — 编译脚本

```bash
nvcc -O3 -std=c++17 -gencode arch=compute_89,code=sm_89 \
    --ptxas-options=-v -allow-unsupported-compiler \
    n_queens_cuda.cu constellation.cpp utils.cpp main.cpp \
    -o n_queens -Xcompiler -fopenmp
```

- `-gencode arch=compute_89,code=sm_89`：目标 RTX 4090（Ada Lovelace 架构）
- `--ptxas-options=-v`：输出寄存器/共享内存统计，用于验证优化
- `-allow-unsupported-compiler`：兼容较新版本的 gcc/nvcc
- `-Xcompiler -fopenmp`：传递给 host 编译器，启用 OpenMP

---

## 三、性能特征

### 实测结果（RTX 4090, preset_queens=6）

| N | 内核时间 | 总耗时 | 解数 | 星座子问题数 |
|---|---------|--------|------|-------------|
| 20 | 2329 ms | 2553 ms | 39,029,188,884 | 1,370,805 |
| 21 | 18632 ms | 18916 ms | 314,666,222,712 | 2,027,630 |
| 22 | 161358 ms | 161725 ms | 2,691,008,701,644 | 2,935,425 |

### 与 NQueensFAF 对比（RTX 4090）

| N | 本求解器 (faf-cuda) | NQueensFAF (OpenCL) | 加速比 |
|---|---------------------|---------------------|--------|
| 20 | 2329 ms | ~2491 ms | 1.07x |
| 21 | 18632 ms | ~18235 ms | 0.98x |
| 22 | 161358 ms | ~157213 ms | 0.97x |

N=20 时本求解器比 NQueensFAF 快约 7%，但 N=21/22 时两者基本持平（NQueensFAF 略快 2-3%）。可能原因：N 增大后共享内存压力上升（jkl_queens 和 queens 栈均随 N 增长），导致每 SM 可容纳的块数减少，影响吞吐量。

### 资源占用

| 指标 | 数值 |
|------|------|
| 寄存器/线程 | 20 |
| 共享内存 (N=20) | 5200 字节 |
| 线程块/SM | 9（共享内存限制） |
| Warp/SM | 18 |

内核是**吞吐量瓶颈**（IPC 限制），RTX 4090 每个 SM 有 4 个 warp 调度器，最大 4 IPC，而需求约 10.8 IPC。增加占用率无效，唯一提升途径是减少每轮迭代的指令数。编译器已生成接近最优的 SASS 代码（LOP3、SHF 指令）。详见 `OPTIMIZATION_LOG.md` 中记录的 10 次失败优化尝试。
