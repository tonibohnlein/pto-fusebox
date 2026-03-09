#pragma once

#include "solution/solution.h"
#include <chrono>

using SteadyClock = std::chrono::steady_clock;
using TimePoint = SteadyClock::time_point;

// No deadline sentinel
inline TimePoint no_deadline() { return TimePoint::max(); }

Solution solve(const Problem& prob, TimePoint deadline = no_deadline());