#include "SteepestDescent.hpp"
#include <cstddef>  // for size_t

SteepestDescent::SteepestDescent(double lr) : Optimizer(lr) {
}

void SteepestDescent::update(
    std::vector<std::vector<double>>& weights,
    std::vector<double>& biases,
    const std::vector<std::vector<double>>& weight_gradients,
    const std::vector<double>& bias_gradients) {
    
    // Get the number of neurons in this layer
    size_t num_neurons = weights.size();
    
    // Update each neuron's parameters
    for (size_t i = 0; i < num_neurons; i++) {
        // Get the number of input connections for this neuron
        size_t num_inputs = weights[i].size();
        
        // Update all weights for this neuron
        // Formula: w_new = w_old - learning_rate * gradient
        for (size_t j = 0; j < num_inputs; j++) {
            weights[i][j] -= learning_rate * weight_gradients[i][j];
        }
        
        // Update bias for this neuron
        // Formula: b_new = b_old - learning_rate * gradient
        biases[i] -= learning_rate * bias_gradients[i];
    }
}