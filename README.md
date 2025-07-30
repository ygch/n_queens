# n_queens
n_queens using openmp and cuda to accelerate.
# Usage
* sh compile.sh
* ./n_queens 16 5
# Run time(ms)
| N  |  CPU |Openmp| 4090 | A100 | 5090 | H800 |   Count     |
|:--:|-----:|-----:|-----:|-----:|-----:|-----:|------------:|
| 13 |    12|     2|    97|   117|   153|   645|        73712|
| 14 |    66|     4|    98|   118|   155|   669|       365596|
| 15 |   410|    20|   127|   124|   157|   714|      2279184|
| 16 |  2285|   110|   132|   132|   165|   757|     14772512|
| 17 | 15188|   756|   136|   164|   180|   803|     95815104|
| 18 |114134|  5834|   245|   345|   258|   808|    666090624|
| 19 |859313| 45275|   870|  1486|   788|  1267|   4968057848|
| 20 |  x   |368940|  5983|  9720|  4721|  5326|  39029188884|
| 21 |  x   |  x   | 48322| 78246| 36322| 38205| 314666222712|
| 22 |  x   |  x   |412338|668218|311139|323001|2691008701644|
1. CPU is AMD 9950x3D;
2. Openmp uses 32 threads on AMD 9950x3D; 
3. single 4090/A100/5090/H800 with pre-placing first 6 rows under configuration2.

With one single GPU card, for solving N <= 22, best parameters are: pre-placing first 5 rows with configuration3.

# Cuda runtimes(s) for larger N
|  N   | 20 | 21  | 22  | 23  | 24 | 25  |
|:----:|---:|----:|----:|----:|---:|----:|
|8 4090|1.83|7.73 |57.59|550.7|5048|48085|
|8 A100|3.34|12.96|94.50|831.0|7996|74893|

* Pre-placing first 6 rows under configuration2.

# Cuda runtime for 26-queens
4.1 days

* 8 RTX 5090 with pre-placing first 6 rows under configuration2.
