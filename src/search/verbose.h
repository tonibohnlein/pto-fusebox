#pragma once

// Thread-local verbose flag. Set to false in parallel worker threads
// to suppress internal logging from local_search_from / fm_outer_loop.
extern thread_local bool g_verbose;