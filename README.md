# n_queens
n_queens using openmp and cuda to accelerate.
# Usage
* sh compile.sh
* ./n_queens 19 6
# Run time(ms)
| N  |  CPU |Openmp| 4090 | A100 | 5090 | H800 |   Count     |
|:--:|-----:|-----:|-----:|-----:|-----:|-----:|------------:|
| 13 |    12|     2|    97|   179|   185|   570|        73712|
| 14 |    66|     4|    98|   187|   197|   572|       365596|
| 15 |   410|    20|   107|   215|   204|   611|      2279184|
| 16 |  2285|   110|   118|   253|   212|   668|     14772512|
| 17 | 15188|   756|   130|   294|   224|   738|     95815104|
| 18 |114134|  5834|   211|   346|   288|   776|    666090624|
| 19 |859313| 45275|   752|  1286|   693|  1189|   4968057848|
| 20 |  x   |368940|  4994|  7663|  3845|  4404|  39029188884|
| 21 |  x   |  x   | 40281| 58902| 29033| 30958| 314666222712|
| 22 |  x   |  x   |344471|503639|245726|261718|2691008701644|
1. CPU is AMD 9950x3D;
2. Openmp uses 32 threads on AMD 9950x3D; 
3. single 4090/A100/5090/H800 with pre-placing first 6 rows under configuration2.

With one single GPU card, for solving N <= 22, best parameters are: pre-placing first 5 rows with configuration3.

# Cuda runtimes(s) for larger N
|  N   | 20 | 21  | 22  | 23  | 24 | 25  |
|:----:|---:|----:|----:|----:|---:|----:|
|8 5090|2.34|5.84 |35.55|302.3|2897|27290|
|8 4090|2.63|6.89 |43.29|364.3|3467|33628|
|8 A100|3.04|10.20|71.35|624.5|6058|56655|

* Pre-placing first 6 rows under configuration2.

# Cuda runtime for 26-queens
3.2 days

* 8 RTX 5090 with pre-placing first 6 rows under configuration2.
