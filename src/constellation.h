// =============================================================================
// constellation.h — D4 对称星座生成器接口
//
// 星座方法（Constellation Method）：
//   在棋盘四条边上预先放置4个"边界皇后"（i,j,k,l），
//   再用CPU回溯放置preset_queens个预设皇后，
//   将剩余行的求解交给GPU位并行回溯。
//
// ijkl编码：(i << 15) | (j << 10) | (k << 5) | l
//   i = 第一行(row 0)的皇后列号
//   j = 最后一行(row N-1)的皇后列号
//   k = 第一列(col 0)的皇后行号
//   l = 最后一列(col N-1)的皇后行号
// =============================================================================

#pragma once

#include <vector>

// 星座子问题：CPU端数据结构
//   ld/rd/col — CPU预设皇后放置后的棋盘状态（位掩码）
//   start_ijkl — 高12位存start（回溯起始行），低20位存ijkl编码
//   solutions — 解数（GPU写出，初始-1）
struct Constellation {
    int ld;
    int rd;
    int col;
    int start_ijkl;
    long long solutions;
};

// ---- ijkl 编解码 ----

// 将4个边界皇后位置编码为ijkl整数
int toIjkl(int i, int j, int k, int l);

// 从ijkl解码出各个边界皇后的位置
int geti(int ijkl);  // 第一行皇后列号
int getj(int ijkl);  // 最后一行皇后列号
int getk(int ijkl);  // 第一列皇后行号
int getl(int ijkl);  // 最后一列皇后行号

// 返回ijkl的低15位（jkl三元组），用于排序分组
int getJkl(int ijkl);

// ---- D4 对称 ----

// 计算D4对称倍数
// 返回8（一般）/4（180度对称）/2（90度对称）
// 最终汇总时 sum += results[i] * symmetry(n, ijkl)
int symmetry(int n, int ijkl);

// ---- 星座生成 ----

// 生成所有星座子问题
// 参数：n — 棋盘大小，preset_queens — CPU预设皇后数
// 返回：Constellation列表，每个元素是一个子问题
std::vector<Constellation> generate_constellations(int n, int preset_queens);
