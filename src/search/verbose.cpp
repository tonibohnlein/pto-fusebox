#include "search/verbose.h"
#include <cstdlib>

static bool init_verbose() {
    const char* v = std::getenv("VERBOSE");
    return v && v[0] != '0';
}

thread_local bool g_verbose = init_verbose();