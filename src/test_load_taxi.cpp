#include "../src/load_taxi.hpp"
#include "../src/point_set.hpp"
#include <iostream>

using namespace std;

int main(void) {
    PointSet ps = get_n_pickups(100, NULL);
    for (int i = 0; i < 100; i++) {
        cout << ps.get_x(i) << ", " << ps.get_y(i) << "\n";
    }
    cout << "Size: " << ps.size << "\n";
    return 0;
}
