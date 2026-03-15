#include <iostream>
#include <cassert>

int add(int a, int b) {
    return a + b;
}

int main() {
    assert(add(2,2) == 4);
    assert(add(1,5) == 6);

    std::cout << "All tests passed\n";
}