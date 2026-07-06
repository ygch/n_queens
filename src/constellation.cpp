// =============================================================================
// constellation.cpp — D4 对称星座生成
//
// 移植自 NQueensFAF 项目的 ConstellationUtils.java 和
// ConstellationsGenerator.java。
//
// 星座方法（Constellation Method）：
//   在棋盘四条边上预先放置4个"边界皇后"：
//     i — 第一行(row 0)的皇后列号
//     j — 最后一行(row N-1)的皇后列号
//     k — 第一列(col 0)的皇后行号
//     l — 最后一列(col N-1)的皇后行号
//   这4个皇后通过对角线约束大幅裁剪搜索空间，
//   剩余行由CPU预设皇后 + GPU回溯完成。
//
// D4对称群：
//   棋盘有8种对称操作（4次旋转 x 2次镜像）。
//   通过旋转消除，只枚举等价类的代表，最终乘以对称倍数(8/4/2)。
//
// ijkl编码：(i << 15) | (j << 10) | (k << 5) | l
//   每个值占5 bit(0-31)，支持N ≤ 32
// =============================================================================

#include "constellation.h"

#include <algorithm>
#include <cstdint>
#include <unordered_set>

// ---- ijkl 编解码工具函数 (移植自 ConstellationUtils.java) ----

// 将4个边界皇后位置编码为ijkl整数
int toIjkl(int i, int j, int k, int l) {
    return (i << 15) | (j << 10) | (k << 5) | l;
}

// 从ijkl解码出各个边界皇后的位置
int geti(int ijkl) { return (ijkl >> 15) & 31; }  // 第一行皇后列号
int getj(int ijkl) { return (ijkl >> 10) & 31; }  // 最后一行皇后列号
int getk(int ijkl) { return (ijkl >> 5) & 31; }   // 第一列皇后行号
int getl(int ijkl) { return ijkl & 31; }           // 最后一列皇后行号

// 返回ijkl的低15位（jkl三元组），用于排序分组
// 相同jkl的星座共享同一组jkl_queens，可复用共享内存
int getJkl(int ijkl) { return ijkl & 0x7FFF; }

// 将ijkl顺时针旋转90度
// 变换关系：i'=N-1-k, j'=N-1-l, k'=j, l'=i
static int rot90(int n, int ijkl) {
    return toIjkl(n - 1 - getk(ijkl), n - 1 - getl(ijkl), getj(ijkl), geti(ijkl));
}

// 左右镜像
// 变换关系：i'=N-1-i, j'=N-1-j, k'=l, l'=k
static int mirvert(int n, int ijkl) {
    return toIjkl(n - 1 - geti(ijkl), n - 1 - getj(ijkl), getl(ijkl), getk(ijkl));
}

// 将星座规范化为"j最小"的表示（D4对称消除的第一步）
// 找到4个边界皇后中离角最近的那个，通过旋转将其放到最后一行(j位置)，
// 再镜像使j靠右。确保每个D4等价类只有一个代表。
static int jAsMin(int n, int ijkl) {
    // 找到离角最近的边界皇后
    int min = std::min(getj(ijkl), n - 1 - getj(ijkl));
    int arg = 0;

    if (std::min(geti(ijkl), n - 1 - geti(ijkl)) < min) {
        arg = 2;  // i离角更近，需旋转2次
        min = std::min(geti(ijkl), n - 1 - geti(ijkl));
    }
    if (std::min(getk(ijkl), n - 1 - getk(ijkl)) < min) {
        arg = 3;  // k离角更近，需旋转3次
        min = std::min(getk(ijkl), n - 1 - getk(ijkl));
    }
    if (std::min(getl(ijkl), n - 1 - getl(ijkl)) < min) {
        arg = 1;  // l离角更近，需旋转1次
        min = std::min(getl(ijkl), n - 1 - getl(ijkl));
    }

    // 旋转arg次，将最近的皇后转到j位置
    for (int i = 0; i < arg; i++)
        ijkl = rot90(n, ijkl);

    // 镜像使j靠右（j > N-1-j时镜像）
    if (getj(ijkl) < n - 1 - getj(ijkl))
        ijkl = mirvert(n, ijkl);

    return ijkl;
}

// 检查(i,j,k,l)的90°/180°/270°旋转是否已存在于集合中
// 若已存在则跳过，避免重复枚举（D4对称消除）
static bool checkRotations(int n, const std::unordered_set<int> &ijklList,
                           int i, int j, int k, int l) {
    // 90度旋转
    if (ijklList.count(toIjkl(n - 1 - k, n - 1 - l, j, i)))
        return true;
    // 180度旋转
    if (ijklList.count(toIjkl(n - 1 - j, n - 1 - i, n - 1 - l, n - 1 - k)))
        return true;
    // 270度旋转
    if (ijklList.count(toIjkl(l, k, n - 1 - i, n - 1 - j)))
        return true;
    return false;
}

// 判断星座是否具有90度旋转对称性
// 即旋转90度后与自身相同
static bool symmetry90(int n, int ijkl) {
    return toIjkl(geti(ijkl), getj(ijkl), getk(ijkl), getl(ijkl)) ==
           toIjkl(n - 1 - getk(ijkl), n - 1 - getl(ijkl), getj(ijkl), geti(ijkl));
}

// 计算D4对称倍数
// 返回8（一般）/4（180度对称）/2（90度对称）
// 最终汇总时 sum += results[i] * symmetry(n, ijkl)
int symmetry(int n, int ijkl) {
    // 检查是否具有180度对称性：i+j==N-1 且 k+l==N-1
    if (geti(ijkl) == n - 1 - getj(ijkl) && getk(ijkl) == n - 1 - getl(ijkl)) {
        if (symmetry90(n, ijkl))
            return 2;  // 90度对称：只有4个等价解而非8个
        return 4;       // 180度对称：只有2个等价解而非8个
    }
    return 8;           // 一般情况：8个等价解
}

// ---- 星座生成器 (移植自 ConstellationsGenerator.java) ----

// 全局状态（避免递归传参过多）
static int g_n;                    // 棋盘大小
static unsigned int g_L;           // 最高位掩码 1<<(N-1)
static unsigned int g_mask;        // N位全1掩码 (1<<N)-1
static int g_presetQueens;         // 预设皇后数
static std::vector<Constellation> g_subConstellations;  // 当前ijkl的子问题列表

// 计算边界皇后j和l产生的左下对角线掩码
// g_L>>j 是j的左对角线，g_L>>l 是l的左对角线
static unsigned int getLD(int ijkl) {
    return (g_L >> getj(ijkl)) | (g_L >> getl(ijkl));
}

// 计算边界皇后j和k产生的右下对角线掩码
// g_L>>j 是j的右对角线，1<<k 是k的右对角线
static unsigned int getRD(int ijkl) {
    return (g_L >> getj(ijkl)) | (1u << getk(ijkl));
}

// 递归回溯：在内部行（非边界行）放置preset_queens个预设皇后
// 参数：
//   ijkl   — 边界皇后配置
//   ld/rd/col — 当前棋盘状态
//   row    — 当前行号
//   queens — 已放置的预设皇后数
static void placePresetQueens(int ijkl, unsigned int ld, unsigned int rd,
                              unsigned int col, int row, int queens) {
    // 跳过边界行（k行和l行已有边界皇后）
    if (row == getk(ijkl) || row == getl(ijkl)) {
        placePresetQueens(ijkl, ld << 1, rd >> 1, col, row + 1, queens);
        return;
    }

    // 放置够preset_queens个皇后，保存当前状态为子问题
    if (queens == g_presetQueens) {
        // start_ijkl = (start行 << 20) | ijkl
        g_subConstellations.push_back(
            {(int)ld, (int)rd, (int)col, (row << 20) | ijkl, -1});
        return;
    }

    // 计算当前行可用位置：
    // ~(ld | rd | col | jkl对角线约束) & N位掩码
    // jkl对角线约束根据行号偏移（从棋盘底部向上延伸）
    unsigned int free =
        (~(ld | rd | col | (getLD(ijkl) >> (g_n - 1 - row)) |
           (getRD(ijkl) << (g_n - 1 - row)))) &
        g_mask;

    // 逐个尝试可用位置（位扫描）
    while (free > 0) {
        unsigned int bit = free & (-free);  // 取最低可用位
        free -= bit;                        // 移除该位
        // 递归：放置皇后，更新状态，进入下一行
        placePresetQueens(ijkl, (ld | bit) << 1, (rd | bit) >> 1, col | bit,
                          row + 1, queens + 1);
    }
}

// 枚举所有不重复的边界皇后配置(i,j,k,l)
static void generateIjkls(std::unordered_set<int> &ijklList) {
    int n = g_n;
    int halfN = (n + 1) / 2;

    // --- 情况1：无角皇后 ---
    // j在半边枚举（利用镜像对称），l在j右侧，k避开l，i避开冲突
    for (int j = 1; j < halfN; j++) {
        for (int l = j + 1; l < n - 1; l++) {
            for (int k = n - j - 2; k > 0; k--) {
                if (k == l)
                    continue;
                for (int i = j + 1; i < n - 1; i++) {
                    if (i == n - 1 - l || i == k)
                        continue;
                    // 用checkRotations去重：跳过旋转等价的配置
                    if (!checkRotations(n, ijklList, i, j, k, l))
                        ijklList.insert(toIjkl(i, j, k, l));
                }
            }
        }
    }

    // --- 情况2：有一个角皇后(0,0) ---
    // 通过D4对称，一个角皇后的情况等价于所有4个角
    for (int k = 1; k < n - 2; k++) {
        for (int i = k + 1; i < n - 1; i++) {
            // j=N-1表示j在角(N-1,N-1)，与i=0的角皇后(0,0)对角对称
            ijklList.insert(toIjkl(i, n - 1, k, n - 1));
        }
    }
}

// 主入口：生成所有星座子问题
// 返回Constellation列表，每个元素是一个子问题（4边界皇后 + preset_queens个预设皇后）
std::vector<Constellation> generate_constellations(int n, int preset_queens) {
    // 设置全局状态
    g_n = n;
    g_L = 1u << (n - 1);
    g_mask = (g_L << 1) - 1;  // N位全1
    g_presetQueens = preset_queens;

    // 生成所有不重复的边界皇后配置
    std::unordered_set<int> ijklList;
    generateIjkls(ijklList);

    // 对每个边界配置，规范化后放置预设皇后
    std::vector<Constellation> constellations;
    for (int ijkl : ijklList) {
        // jAsMin规范化：通过旋转/镜像将星座转为标准形式
        ijkl = jAsMin(n, ijkl);
        int i = geti(ijkl), j = getj(ijkl), k = getk(ijkl), l = getl(ijkl);

        // 计算4个边界皇后的初始ld/rd/col状态
        // i皇后在第一行：左对角线L>>(i-1)，右对角线L>>(i+1)
        unsigned int ld = (g_L >> (i - 1)) | (1u << (n - k));
        unsigned int rd = (g_L >> (i + 1)) | (1u << (l - 1));
        // col：4个边界皇后的列占用 + 两个边界列
        unsigned int col = 1 | g_L | (g_L >> i) | (g_L >> j);

        // 放置预设皇后，生成子问题
        // 初始预设皇后数：j==N-1时为3（角皇后情况，已有4个边界但j在角上），
        //                  否则为4（4个边界皇后各占一行）
        g_subConstellations.clear();
        placePresetQueens(ijkl, ld, rd, col, 1, j == n - 1 ? 3 : 4);

        for (auto &c : g_subConstellations)
            constellations.push_back(c);
    }

    return constellations;
}
