# n_queens
n_queens using openmp and cuda to accelerate.
# Usage
* sh compile.sh
* ./n_queens 19 6
# Differences between versions
* v1.0: Basic implementation, plus int4 optimization;
* v2.0: Inline PTX optimization;
* v2.1: Dynamic task fetching optimization;
* v3.0: Further inline PTX optimization;

The following results are achieved based on v2.1. v3.0 has a slight improvement, can roughly bring about 1%~4% speedup.
# Run time(ms)
| N  |  CPU |Openmp| 4090 | A100 | 5090 | H800 |   Count     |
|:--:|-----:|-----:|-----:|-----:|-----:|-----:|------------:|
| 13 |    12|     2|   107|   123|   185|   558|        73712|
| 14 |    66|     4|   108|   166|   193|   614|       365596|
| 15 |   410|    20|   110|   174|   197|   627|      2279184|
| 16 |  2285|   110|   116|   209|   202|   645|     14772512|
| 17 | 15188|   756|   126|   247|   209|   670|     95815104|
| 18 |114134|  5834|   183|   267|   261|   791|    666090624|
| 19 |859313| 45275|   609|  1074|   557|  1138|   4968057848|
| 20 |  x   |368940|  3636|  6415|  2778|  4140|  39029188884|
| 21 |  x   |  x   | 28551| 51741| 21242| 29248| 314666222712|
| 22 |  x   |  x   |245090|444272|180364|249886|2691008701644|
1. CPU is AMD 9950x3D;
2. Openmp uses 32 threads on AMD 9950x3D; 
3. single 4090/A100/5090/H800 with pre-placing first 6 rows under configuration2.

With one single GPU card, for solving N <= 22, best parameters are: pre-placing first 5 rows with configuration3.

# Cuda runtimes(s) for larger N
|  N   | 20 | 21  | 22  | 23  | 24 | 25  |
|:----:|---:|----:|----:|----:|---:|----:|
|8 5090|2.15|4.91 |29.10|232.7|2205|21880|
|8 4090|2.58|6.16 |37.46|308.5|2840|28332|
|8 A100|2.89|9.24 |67.58|562.0|5307|53104|

* Pre-placing first 6 rows under configuration2.

# Cuda runtime for 26-queens
2.5 days
* 8 RTX 5090 with pre-placing first 6 rows under configuration2.

# Cuda runtime for 27-queens
28.4 days

* 8 RTX 5090 with pre-placing first 7 rows under configuration2.

# Projected cuda runtime for 28-queens
340 days = 28.4 * 10.52 * 1.14

* 8 RTX 5090 with pre-placing first 6 rows under configuration1.

# Citation
If you find our paper and code useful in your research, please consider giving a star ⭐ and citation 📝 :)

```BibTeX
@article{GPU-N-queens,
  title={High-Performance N-Queens Solver on GPU: Iterative DFS with Zero Bank Conflicts},
  author={Guangchao Yao, Yali Li},
  journal={arXiv preprint arXiv:2511.12009},
  year={2025}
}
```
Link: https://arxiv.org/pdf/2511.12009
