#pragma once

#include <vector>

struct Constellation {
    int ld;
    int rd;
    int col;
    int start_ijkl;
    long long solutions;
};

// ijkl encoding: (i << 15) | (j << 10) | (k << 5) | l
// i = queen in first row, j = queen in last row
// k = queen in first col, l = queen in last col
int toIjkl(int i, int j, int k, int l);
int geti(int ijkl);
int getj(int ijkl);
int getk(int ijkl);
int getl(int ijkl);
int getJkl(int ijkl);

// D4 symmetry multiplier for a border configuration
int symmetry(int n, int ijkl);

// Generate all subproblems for the D4 constellation method
std::vector<Constellation> generate_constellations(int n, int preset_queens);
