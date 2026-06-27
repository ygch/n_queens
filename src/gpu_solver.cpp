#include "gpu_solver.h"
#include "constellation.h"
#include "utils.h"

#include <CL/cl.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define WG_SIZE 64

static const char *kernel_source = R"(
struct constellation {
    int ld;
    int rd;
    int col;
    int start_ijkl;
};

kernel void nqfaf_nvidia(global struct constellation *constellation_arr,
                         global uint *jkl_queens_arr,
                         global long *result) {
    const struct constellation c = constellation_arr[get_global_id(0)];

    const int start = c.start_ijkl >> 20 & 31;
    if (start == 69)
        return;

    const int l_id = get_local_id(0);
    const uint L = 1 << (N - 1);

    uint ld = c.ld;
    uint rd = c.rd;
    uint col = ~(L - 2) ^ c.col;
    uint ld_mem = 0;
    uint rd_mem = 0;

    local uint jkl_queens[N];
    jkl_queens[l_id % N] = jkl_queens_arr[get_group_id(0) * N + l_id % N];
    barrier(CLK_LOCAL_MEM_FENCE);

    ld &= ~(L >> (((c.start_ijkl >> 5) & 31) - start));
    if ((c.start_ijkl & 31) != N - 1)
        rd &= ~(1 << ((c.start_ijkl & 31) - start));

    int row = start;
    ulong solutions = 0;

    uint free = ~(ld | rd | col | jkl_queens[row]);
    uint queen = -free & free;

    local uint queens[WORKGROUP_SIZE][N];
    queens[l_id][start] = queen;

    int direction = 0;

    while (row >= start) {
        if (free) {
            if (row == N - 2) {
                solutions++;
                direction = 0;
                row--;
                queen = queens[l_id][row];
                ld = ((ld >> 1) | (ld_mem << 31)) & ~queen;
                rd = ((rd << 1) | (rd_mem >> 31)) & ~queen;
                ld_mem >>= 1;
                rd_mem <<= 1;
            } else {
                direction = 1;
                queen = -free & free;
                queens[l_id][row] = queen;
                row++;
                ld_mem = ld_mem << 1 | ld >> 31;
                rd_mem = rd_mem >> 1 | rd << 31;
                ld = (ld | queen) << 1;
                rd = (rd | queen) >> 1;
            }
        } else {
            direction = 0;
            row--;
            queen = queens[l_id][row];
            ld = ((ld >> 1) | (ld_mem << 31)) & ~queen;
            rd = ((rd << 1) | (rd_mem >> 31)) & ~queen;
            ld_mem >>= 1;
            rd_mem <<= 1;
        }
        free = ~(jkl_queens[row] | ld | rd | col);
        free &= ~(queen + direction - 1);
        col ^= queen;

        if (row < start)
            break;

        if (free) {
            if (row == N - 2) {
                solutions++;
                direction = 0;
                row--;
                queen = queens[l_id][row];
                ld = ((ld >> 1) | (ld_mem << 31)) & ~queen;
                rd = ((rd << 1) | (rd_mem >> 31)) & ~queen;
                ld_mem >>= 1;
                rd_mem <<= 1;
            } else {
                direction = 1;
                queen = -free & free;
                queens[l_id][row] = queen;
                row++;
                ld_mem = ld_mem << 1 | ld >> 31;
                rd_mem = rd_mem >> 1 | rd << 31;
                ld = (ld | queen) << 1;
                rd = (rd | queen) >> 1;
            }
        } else {
            direction = 0;
            row--;
            queen = queens[l_id][row];
            ld = ((ld >> 1) | (ld_mem << 31)) & ~queen;
            rd = ((rd << 1) | (rd_mem >> 31)) & ~queen;
            ld_mem >>= 1;
            rd_mem <<= 1;
        }
        free = ~(jkl_queens[row] | ld | rd | col);
        free &= ~(queen + direction - 1);
        col ^= queen;
    }
    result[get_global_id(0)] = solutions;
}
)";

static void checkCLError(cl_int err, const char *msg) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL error %d: %s\n", err, msg);
        exit(-1);
    }
}

// sort constellations by jkl so each workgroup shares the same jkl
static void sortByJkl(std::vector<Constellation> &consts) {
    std::sort(consts.begin(), consts.end(), [](const Constellation &a, const Constellation &b) {
        return getJkl(a.start_ijkl) < getJkl(b.start_ijkl);
    });
}

// pad each jkl group to a multiple of workgroup size with pseudo constellations
static void padWithPseudo(std::vector<Constellation> &consts, int n, int wgSize) {
    sortByJkl(consts);
    std::vector<Constellation> padded;
    int allOnes = (1 << n) - 1;
    int currentJkl = getJkl(consts[0].start_ijkl);
    for (auto &c : consts) {
        if (getJkl(c.start_ijkl) != currentJkl) {
            while (padded.size() % wgSize != 0)
                padded.push_back({allOnes, allOnes, allOnes, (69 << 20), -2});
            currentJkl = getJkl(c.start_ijkl);
        }
        padded.push_back(c);
    }
    while (padded.size() % wgSize != 0)
        padded.push_back({allOnes, allOnes, allOnes, (69 << 20), -2});
    consts = std::move(padded);
}

long long opencl_solve(int n, int preset_queens) {
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    auto constellations = generate_constellations(n, preset_queens);
    gettimeofday(&t1, NULL);
    print_with_time("Constellation generation: %.2fms, %zu subproblems\n",
                    time_diff_ms(t0, t1), constellations.size());

    if (constellations.empty()) {
        print_with_time("No subproblems generated!\n");
        return 0;
    }

    padWithPseudo(constellations, n, WG_SIZE);
    long long cnt = constellations.size();
    int numWg = cnt / WG_SIZE;

    print_with_time("Total constellations (padded): %lld, workgroups: %d\n", cnt, numWg);

    // ---- precompute jkl_queens for each workgroup ----
    unsigned int L = 1u << (n - 1);
    std::vector<uint32_t> jkl_queens_arr(numWg * n, 0);
    for (int wg = 0; wg < numWg; wg++) {
        int ijkl = constellations[wg * WG_SIZE].start_ijkl & 0xFFFFF;
        int j = getj(ijkl), k = getk(ijkl), l = getl(ijkl);
        unsigned int rdiag = (L >> j) | (L >> (n - 1 - k));
        unsigned int ldiag = (L >> j) | (L >> l);
        for (int row = 0; row < n; row++)
            jkl_queens_arr[wg * n + (n - 1 - row)] = (ldiag >> row) | (rdiag << row) | L | 1;
        ldiag = L >> k;
        rdiag = 1u << l;
        for (int row = 0; row < n; row++)
            jkl_queens_arr[wg * n + row] |= (ldiag << row) | (rdiag >> row);
        jkl_queens_arr[wg * n + k] = ~L;
        jkl_queens_arr[wg * n + l] = ~1u;
    }

    // ---- flatten constellation data ----
    std::vector<int> const_data(cnt * 4);
    for (long long i = 0; i < cnt; i++) {
        const_data[i * 4 + 0] = constellations[i].ld;
        const_data[i * 4 + 1] = constellations[i].rd;
        const_data[i * 4 + 2] = constellations[i].col;
        const_data[i * 4 + 3] = constellations[i].start_ijkl;
    }

    // ---- OpenCL setup ----
    cl_platform_id platform;
    cl_uint numPlatforms;
    clGetPlatformIDs(1, &platform, &numPlatforms);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    char deviceName[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, 256, deviceName, NULL);
    print_with_time("Using device: %s\n", deviceName);

    cl_int err;
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkCLError(err, "clCreateContext");

    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    checkCLError(err, "clCreateProgramWithSource");

    char buildOpts[128];
    snprintf(buildOpts, sizeof(buildOpts), "-D N=%d -D WORKGROUP_SIZE=%d -Werror", n, WG_SIZE);
    err = clBuildProgram(program, 1, &device, buildOpts, NULL, NULL);
    if (err != CL_SUCCESS) {
        char buildLog[4096];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
        fprintf(stderr, "Build error:\n%s\n", buildLog);
        exit(-1);
    }

    cl_kernel kernel = clCreateKernel(program, "nqfaf_nvidia", &err);
    checkCLError(err, "clCreateKernel");

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    checkCLError(err, "clCreateCommandQueue");

    // ---- create buffers ----
    cl_mem constBuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    cnt * 4 * sizeof(int), const_data.data(), &err);
    checkCLError(err, "clCreateBuffer constBuf");

    cl_mem jklBuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   numWg * n * sizeof(uint32_t), jkl_queens_arr.data(), &err);
    checkCLError(err, "clCreateBuffer jklBuf");

    std::vector<long long> results(cnt, 0);
    cl_mem resBuf = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
                                  cnt * sizeof(long long), results.data(), &err);
    checkCLError(err, "clCreateBuffer resBuf");

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &constBuf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &jklBuf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &resBuf);

    // ---- dispatch kernel ----
    gettimeofday(&t0, NULL);
    size_t globalSize = cnt;
    size_t localSize = WG_SIZE;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
    checkCLError(err, "clEnqueueNDRangeKernel");
    clFinish(queue);
    gettimeofday(&t1, NULL);
    print_with_time("OpenCL kernel time: %.2fms\n", time_diff_ms(t0, t1));

    // ---- read results ----
    clEnqueueReadBuffer(queue, resBuf, CL_TRUE, 0, cnt * sizeof(long long), results.data(), 0, NULL, NULL);

    // ---- sum with symmetry multipliers ----
    long long sum = 0;
    for (long long i = 0; i < cnt; i++) {
        if (constellations[i].start_ijkl >> 20 == 69)
            continue; // pseudo constellation
        int ijkl = constellations[i].start_ijkl & 0xFFFFF;
        sum += results[i] * symmetry(n, ijkl);
    }

    // ---- cleanup ----
    clReleaseMemObject(constBuf);
    clReleaseMemObject(jklBuf);
    clReleaseMemObject(resBuf);
    clReleaseCommandQueue(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);

    return sum;
}
