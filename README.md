# n_queens
n_queens using openmp and cuda to accelerate.
# Usage
* sh compile.sh
* ./n_queens 16 5
# Run time(ms)
| N | CPU | Openmp | 4090 | A100|    Count|
|:-:|----:|-------:|-----:|-------:|--------:|
|  13 |    12|     2|    97|   117|     73712|
|  14 |    66|     4|    98|   118|    365596|
|  15 |   410|    20|   127|   124|   2279184|
|  16 |  2285|   110|   132|   132|  14772512|
|  17 | 15188|   756|   136|   164|  95815104|
|  18 |114134|  5834|   245|   345| 666090624|
|  19 |  x   | 45275|   870|  1486| 4968057848|
|  20 |  x   |368940|  5983|  9720|39029188884|
|  21 |  x   |  x   | 48322| 78246|314666222712|
|  22 |  x   |  x   |412338|668218|2691008701644|
## Comments
1. CPU is AMD 9950x3D;
2. Openmp uses 32 threads on AMD 9950x3D; 
3. single 4090 with pre-placing first 6 rows;
4. single A100 with pre-placing first 6 rows;

# Cuda runtimes(s) for larger N
| N | 20 | 21 | 22 | 23 | 24 | 25 |
|:-:|---:|---:|---:|---:|---:|---:|
|Time|1.86|7.92|59.72|550.5|5048|48085|
## Comments
* 8 RTX 4090 with pre-placing first 6 rows;

# Cuda runtime for 26-queens
4.1 days

## Comments
* 8 RTX 5090 with pre-placing first 6 rows;
