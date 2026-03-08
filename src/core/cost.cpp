#include "core/cost.h"

std::vector<int> make_traversal(int num_tw, int num_th, SnakeDir dir) {
    std::vector<int> order;
    order.reserve(num_tw * num_th);

    if (dir == SnakeDir::ColMajor) {
        for (int c = 0; c < num_tw; c++) {
            if (c % 2 == 1)
                for (int r = num_th - 1; r >= 0; r--) order.push_back(r * num_tw + c);
            else
                for (int r = 0; r < num_th; r++) order.push_back(r * num_tw + c);
        }
    } else {
        for (int r = 0; r < num_th; r++) {
            if (dir == SnakeDir::RowMajor && r % 2 == 1)
                for (int c = num_tw - 1; c >= 0; c--) order.push_back(r * num_tw + c);
            else
                for (int c = 0; c < num_tw; c++) order.push_back(r * num_tw + c);
        }
    }
    return order;
}
