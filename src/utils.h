#pragma once

#include <sys/time.h>

float time_diff_ms(struct timeval &start, struct timeval &end);

void print_with_time(const char *format, ...);
