#include <iostream>
#include <vector>
#include "Network.hpp"

int main()
{
    // XOR dataset
    std::vector<std::vector<double>> X = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    std::vector<std::vector<double>> Y = {
        {0},
        {1},
        {1},
        {0}
    };

    // Create network: 2 inputs, 3 hidden, 1 output
    Network net(
        {2, 3, 1},
        Activations::sigmoid(),
        OptimizerType::STEEPEST_DESCENT,  // najjednostavniji optimizer
        0.5                                // learning rate
    );

    // Train
    net.fit(X, Y, 5000, true);

    // Test predictions
    std::cout << "\n--- Predictions after training ---\n";
    for (size_t i = 0; i < X.size(); ++i)
    {
        auto out = net.predict(X[i]);
        std::cout << X[i][0] << " XOR " << X[i][1]
                  << " = " << out[0] << std::endl;
    }

    return 0;
}
