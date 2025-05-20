#include "utils.h"

#include <chrono>
#include <cstdarg>
#include <iomanip>
#include <sstream>

using namespace std;

float time_diff_ms(struct timeval &start, struct timeval &end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
}

void print_with_time(const char *format, ...) {
    auto now = chrono::system_clock::now();
    auto ms = chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch()) % 1000;

    auto now_time = chrono::system_clock::to_time_t(now);
    tm tm = *localtime(&now_time);

    stringstream time_ss;
    time_ss << put_time(&tm, "[%Y-%m-%d %H:%M:%S.") << setfill('0') << setw(3) << ms.count() << "] ";

    fputs(time_ss.str().c_str(), stdout);

    va_list args;
    va_start(args, format);
    vfprintf(stdout, format, args);
    va_end(args);
}
