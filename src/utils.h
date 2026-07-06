// =============================================================================
// utils.h — 工具函数接口
// =============================================================================

#pragma once

#include <sys/time.h>

// 计算两个timeval之间的毫秒差
float time_diff_ms(struct timeval &start, struct timeval &end);

// 带时间戳的printf，格式：[YYYY-MM-DD HH:MM:SS.mmm] message
void print_with_time(const char *format, ...);
