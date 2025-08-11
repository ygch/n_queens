# n_queens
n_queens using openmp and cuda to accelerate.
# Usage
* sh compile.sh
* ./n_queens 19 6
# Run time(ms)
| N  |  CPU |Openmp| 4090 | A100 | 5090 | H800 |   Count     |
|:--:|-----:|-----:|-----:|-----:|-----:|-----:|------------:|
| 13 |    12|     2|   107|   123|   185|   558|        73712|
| 14 |    66|     4|   108|   166|   193|   614|       365596|
| 15 |   410|    20|   110|   174|   197|   627|      2279184|
| 16 |  2285|   110|   116|   209|   202|   645|     14772512|
| 17 | 15188|   756|   126|   247|   209|   670|     95815104|
| 18 |114134|  5834|   183|   328|   282|   861|    666090624|
| 19 |859313| 45275|   610|  1133|   614|  1108|   4968057848|
| 20 |  x   |368940|  3978|  7027|  3274|  4427|  39029188884|
| 21 |  x   |  x   | 31836| 56797| 24425| 33042| 314666222712|
| 22 |  x   |  x   |269498|485752|206644|270142|2691008701644|
1. CPU is AMD 9950x3D;
2. Openmp uses 32 threads on AMD 9950x3D; 
3. single 4090/A100/5090/H800 with pre-placing first 6 rows under configuration2.

With one single GPU card, for solving N <= 22, best parameters are: pre-placing first 5 rows with configuration3.

# Cuda runtimes(s) for larger N
|  N   | 20 | 21  | 22  | 23  | 24 | 25  |
|:----:|---:|----:|----:|----:|---:|----:|
|8 5090|2.24|5.17 |30.30|258.5|2474|23432|
|8 4090|2.58|6.42 |38.46|337.9|3254|30378|
|8 A100|2.96|10.30|69.90|620.5|5846|54732|

* Pre-placing first 6 rows under configuration2.

# Cuda runtime for 26-queens
2.7 days

* 8 RTX 5090 with pre-placing first 6 rows under configuration2.
