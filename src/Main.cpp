#include <iostream>
#include "Network.hpp"

int main() {
    Network net({2, 3, 1}, Activations::sigmoid(), 0.1);

    std::vector<double> x{0, 1};
    std::vector<double> y{1};

    net.train(x, y);

    auto out = net.predict(x);
    std::cout << "out = " << out[0] << "\n";
    return 0;
}