#include "constellation.h"

#include <algorithm>
#include <cstdint>
#include <unordered_set>

// ---- ijkl utilities (ported from ConstellationUtils.java) ----

int toIjkl(int i, int j, int k, int l) {
    return (i << 15) | (j << 10) | (k << 5) | l;
}
int geti(int ijkl) { return (ijkl >> 15) & 31; }
int getj(int ijkl) { return (ijkl >> 10) & 31; }
int getk(int ijkl) { return (ijkl >> 5) & 31; }
int getl(int ijkl) { return ijkl & 31; }
int getJkl(int ijkl) { return ijkl & 0x7FFF; }

// rotate ijkl 90 degrees clockwise
static int rot90(int n, int ijkl) {
    return toIjkl(n - 1 - getk(ijkl), n - 1 - getl(ijkl), getj(ijkl), geti(ijkl));
}

// mirror left-right
static int mirvert(int n, int ijkl) {
    return toIjkl(n - 1 - geti(ijkl), n - 1 - getj(ijkl), getl(ijkl), getk(ijkl));
}

// rotate and mirror so the queen closest to a corner is on the right side of the last row
static int jAsMin(int n, int ijkl) {
    int min = std::min(getj(ijkl), n - 1 - getj(ijkl));
    int arg = 0;

    if (std::min(geti(ijkl), n - 1 - geti(ijkl)) < min) {
        arg = 2;
        min = std::min(geti(ijkl), n - 1 - geti(ijkl));
    }
    if (std::min(getk(ijkl), n - 1 - getk(ijkl)) < min) {
        arg = 3;
        min = std::min(getk(ijkl), n - 1 - getk(ijkl));
    }
    if (std::min(getl(ijkl), n - 1 - getl(ijkl)) < min) {
        arg = 1;
        min = std::min(getl(ijkl), n - 1 - getl(ijkl));
    }

    for (int i = 0; i < arg; i++)
        ijkl = rot90(n, ijkl);

    if (getj(ijkl) < n - 1 - getj(ijkl))
        ijkl = mirvert(n, ijkl);

    return ijkl;
}

// true if any 90/180/270 rotation of (i,j,k,l) is already in the set
static bool checkRotations(int n, const std::unordered_set<int> &ijklList,
                           int i, int j, int k, int l) {
    if (ijklList.count(toIjkl(n - 1 - k, n - 1 - l, j, i)))
        return true;
    if (ijklList.count(toIjkl(n - 1 - j, n - 1 - i, n - 1 - l, n - 1 - k)))
        return true;
    if (ijklList.count(toIjkl(l, k, n - 1 - i, n - 1 - j)))
        return true;
    return false;
}

// true if the configuration is symmetric under 90-degree rotation
static bool symmetry90(int n, int ijkl) {
    return toIjkl(geti(ijkl), getj(ijkl), getk(ijkl), getl(ijkl)) ==
           toIjkl(n - 1 - getk(ijkl), n - 1 - getl(ijkl), getj(ijkl), geti(ijkl));
}

int symmetry(int n, int ijkl) {
    if (geti(ijkl) == n - 1 - getj(ijkl) && getk(ijkl) == n - 1 - getl(ijkl)) {
        if (symmetry90(n, ijkl))
            return 2;
        return 4;
    }
    return 8;
}

// ---- Constellation generator (ported from ConstellationsGenerator.java) ----

static int g_n;
static unsigned int g_L;
static unsigned int g_mask;
static int g_presetQueens;
static std::vector<Constellation> g_subConstellations;

// bottom-left diagonal from queens j and l
static unsigned int getLD(int ijkl) {
    return (g_L >> getj(ijkl)) | (g_L >> getl(ijkl));
}
// bottom-right diagonal from queens j and k
static unsigned int getRD(int ijkl) {
    return (g_L >> getj(ijkl)) | (1u << getk(ijkl));
}

static void placePresetQueens(int ijkl, unsigned int ld, unsigned int rd,
                              unsigned int col, int row, int queens) {
    if (row == getk(ijkl) || row == getl(ijkl)) {
        placePresetQueens(ijkl, ld << 1, rd >> 1, col, row + 1, queens);
        return;
    }
    if (queens == g_presetQueens) {
        g_subConstellations.push_back(
            {(int)ld, (int)rd, (int)col, (row << 20) | ijkl, -1});
        return;
    }
    unsigned int free =
        (~(ld | rd | col | (getLD(ijkl) >> (g_n - 1 - row)) |
           (getRD(ijkl) << (g_n - 1 - row)))) &
        g_mask;
    while (free > 0) {
        unsigned int bit = free & (-free);
        free -= bit;
        placePresetQueens(ijkl, (ld | bit) << 1, (rd | bit) >> 1, col | bit,
                          row + 1, queens + 1);
    }
}

static void generateIjkls(std::unordered_set<int> &ijklList) {
    int n = g_n;
    int halfN = (n + 1) / 2;

    // no queens in corners
    for (int j = 1; j < halfN; j++) {
        for (int l = j + 1; l < n - 1; l++) {
            for (int k = n - j - 2; k > 0; k--) {
                if (k == l)
                    continue;
                for (int i = j + 1; i < n - 1; i++) {
                    if (i == n - 1 - l || i == k)
                        continue;
                    if (!checkRotations(n, ijklList, i, j, k, l))
                        ijklList.insert(toIjkl(i, j, k, l));
                }
            }
        }
    }

    // one queen in corner (0,0) — by D4 rotation, equivalent to corner (N-1,N-1)
    for (int k = 1; k < n - 2; k++) {
        for (int i = k + 1; i < n - 1; i++) {
            ijklList.insert(toIjkl(i, n - 1, k, n - 1));
        }
    }
}

std::vector<Constellation> generate_constellations(int n, int preset_queens) {
    g_n = n;
    g_L = 1u << (n - 1);
    g_mask = (g_L << 1) - 1;
    g_presetQueens = preset_queens;

    std::unordered_set<int> ijklList;
    generateIjkls(ijklList);

    std::vector<Constellation> constellations;
    for (int ijkl : ijklList) {
        ijkl = jAsMin(n, ijkl);
        int i = geti(ijkl), j = getj(ijkl), k = getk(ijkl), l = getl(ijkl);

        unsigned int ld = (g_L >> (i - 1)) | (1u << (n - k));
        unsigned int rd = (g_L >> (i + 1)) | (1u << (l - 1));
        unsigned int col = 1 | g_L | (g_L >> i) | (g_L >> j);

        g_subConstellations.clear();
        placePresetQueens(ijkl, ld, rd, col, 1, j == n - 1 ? 3 : 4);

        for (auto &c : g_subConstellations)
            constellations.push_back(c);
    }

    return constellations;
}
