// =============================================================================
// utils.cpp — 工具函数实现
// =============================================================================

#include "utils.h"

#include <chrono>
#include <cstdarg>
#include <iomanip>
#include <sstream>

using namespace std;

// 计算两个timeval之间的毫秒差
float time_diff_ms(struct timeval &start, struct timeval &end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
}

// 带时间戳的printf
// 输出格式：[YYYY-MM-DD HH:MM:SS.mmm] <用户消息>
void print_with_time(const char *format, ...) {
    auto now = chrono::system_clock::now();
    auto ms = chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch()) % 1000;

    auto now_time = chrono::system_clock::to_time_t(now);
    tm tm = *localtime(&now_time);

    // 格式化时间戳前缀
    stringstream time_ss;
    time_ss << put_time(&tm, "[%Y-%m-%d %H:%M:%S.") << setfill('0') << setw(3) << ms.count() << "] ";

    fputs(time_ss.str().c_str(), stdout);

    // 输出用户消息
    va_list args;
    va_start(args, format);
    vfprintf(stdout, format, args);
    va_end(args);
}
