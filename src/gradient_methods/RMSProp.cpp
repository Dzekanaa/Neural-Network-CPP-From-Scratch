#include "../include/gradient_methods/RMSProp.hpp"
#include <cmath>
#include <cstddef>

RMSProp::RMSProp(double lr, double decay, double eps)
    : Optimizer(lr), decay_rate(decay), epsilon(eps), initialized(false) {
}

void RMSProp::initialize_cache(size_t num_neurons, size_t num_inputs) {
    // Initialize weight cache
    weight_cache.resize(num_neurons, std::vector<double>(num_inputs, 0.0));
    
    // Initialize bias cache
    bias_cache.resize(num_neurons, 0.0);
    
    initialized = true;
}

void RMSProp::update(
    std::vector<std::vector<double>>& weights,
    std::vector<double>& biases,
    const std::vector<std::vector<double>>& weight_gradients,
    const std::vector<double>& bias_gradients) {
    
    size_t num_neurons = weights.size();
    
    if (!initialized && num_neurons > 0) {
        size_t num_inputs = weights[0].size();
        initialize_cache(num_neurons, num_inputs);
    }
    
    // Update each neuron's parameters
    for (size_t i = 0; i < num_neurons; i++) {
        size_t num_inputs = weights[i].size();
        
        // Update weights
        for (size_t j = 0; j < num_inputs; j++) {
            // Update moving average of squared gradients
            // cache = decay * cache + (1 - decay) * gradient^2
            weight_cache[i][j] = decay_rate * weight_cache[i][j] 
                               + (1.0 - decay_rate) * weight_gradients[i][j] * weight_gradients[i][j];
            
            // Update weight with adaptive learning rate
            // w = w - (lr / sqrt(cache + epsilon)) * gradient
            weights[i][j] -= (learning_rate / std::sqrt(weight_cache[i][j] + epsilon)) 
                            * weight_gradients[i][j];
        }
        
        // Update bias
        // Update moving average of squared gradients
        bias_cache[i] = decay_rate * bias_cache[i] 
                       + (1.0 - decay_rate) * bias_gradients[i] * bias_gradients[i];
        
        // Update bias with adaptive learning rate
        biases[i] -= (learning_rate / std::sqrt(bias_cache[i] + epsilon)) 
                    * bias_gradients[i];
    }
}

void RMSProp::reset() {
    // Clear all moving averages
    for (auto& row : weight_cache) {
        std::fill(row.begin(), row.end(), 0.0);
    }
    std::fill(bias_cache.begin(), bias_cache.end(), 0.0);
}

void RMSProp::set_decay_rate(double decay) {
    decay_rate = decay;
}