#include "ADAGRAD.hpp"
#include <cmath>
#include <cstddef>

ADAGRAD::ADAGRAD(double lr, double eps)
    : Optimizer(lr), epsilon(eps), initialized(false) {
}

void ADAGRAD::initialize_cache(size_t num_neurons, size_t num_inputs) {
    // Initialize weight cache
    weight_cache.resize(num_neurons, std::vector<double>(num_inputs, 0.0));
    
    // Initialize bias cache
    bias_cache.resize(num_neurons, 0.0);
    
    initialized = true;
}

void ADAGRAD::update(
    std::vector<std::vector<double>>& weights,
    std::vector<double>& biases,
    const std::vector<std::vector<double>>& weight_gradients,
    const std::vector<double>& bias_gradients) {
    
    size_t num_neurons = weights.size();
    
    // Initialize cache on first call
    if (!initialized && num_neurons > 0) {
        size_t num_inputs = weights[0].size();
        initialize_cache(num_neurons, num_inputs);
    }
    
    // Update each neuron's parameters
    for (size_t i = 0; i < num_neurons; i++) {
        size_t num_inputs = weights[i].size();
        
        // Update weights
        for (size_t j = 0; j < num_inputs; j++) {
            // Accumulate squared gradient
            weight_cache[i][j] += weight_gradients[i][j] * weight_gradients[i][j];
            
            // Update weight with adaptive learning rate
            // w = w - (lr / sqrt(cache + epsilon)) * gradient
            weights[i][j] -= (learning_rate / std::sqrt(weight_cache[i][j] + epsilon)) 
                            * weight_gradients[i][j];
        }
        
        // Update bias
        // Accumulate squared gradient
        bias_cache[i] += bias_gradients[i] * bias_gradients[i];
        
        // Update bias with adaptive learning rate
        biases[i] -= (learning_rate / std::sqrt(bias_cache[i] + epsilon)) 
                    * bias_gradients[i];
    }
}

void ADAGRAD::reset() {
    // Clear all accumulated gradients
    for (auto& row : weight_cache) {
        std::fill(row.begin(), row.end(), 0.0);
    }
    std::fill(bias_cache.begin(), bias_cache.end(), 0.0);
}