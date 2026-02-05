#include "../include/gradient_methods/ADAM.hpp"
#include <cmath>
#include <cstddef>

ADAM::ADAM(double lr, double b1, double b2, double eps)
    : Optimizer(lr), beta1(b1), beta2(b2), epsilon(eps), timestep(0), initialized(false) {
}

void ADAM::initialize_moments(size_t num_neurons, size_t num_inputs) {
    // Initialize first moment (m) for weights and biases
    weight_m.resize(num_neurons, std::vector<double>(num_inputs, 0.0));
    bias_m.resize(num_neurons, 0.0);
    
    // Initialize second moment (v) for weights and biases
    weight_v.resize(num_neurons, std::vector<double>(num_inputs, 0.0));
    bias_v.resize(num_neurons, 0.0);
    
    initialized = true;
}

void ADAM::update(
    std::vector<std::vector<double>>& weights,
    std::vector<double>& biases,
    const std::vector<std::vector<double>>& weight_gradients,
    const std::vector<double>& bias_gradients) {
    
    size_t num_neurons = weights.size();
    
    if (!initialized && num_neurons > 0) {
        size_t num_inputs = weights[0].size();
        initialize_moments(num_neurons, num_inputs);
    }
    
    timestep++;
    
    // Compute bias correction terms
    double bias_correction1 = 1.0 - std::pow(beta1, timestep);
    double bias_correction2 = 1.0 - std::pow(beta2, timestep);
    
    // Update each neuron's parameters
    for (size_t i = 0; i < num_neurons; i++) {
        size_t num_inputs = weights[i].size();
        
        // Update weights
        for (size_t j = 0; j < num_inputs; j++) {
            // Update biased first moment estimate (momentum)
            // m = beta1 * m + (1 - beta1) * gradient
            weight_m[i][j] = beta1 * weight_m[i][j] 
                           + (1.0 - beta1) * weight_gradients[i][j];
            
            // Update biased second moment estimate (RMSProp)
            // v = beta2 * v + (1 - beta2) * gradient^2
            weight_v[i][j] = beta2 * weight_v[i][j] 
                           + (1.0 - beta2) * weight_gradients[i][j] * weight_gradients[i][j];
            
            // Compute bias-corrected moments
            double m_hat = weight_m[i][j] / bias_correction1;
            double v_hat = weight_v[i][j] / bias_correction2;
            
            // Update weight
            // w = w - (lr * m_hat) / (sqrt(v_hat) + epsilon)
            weights[i][j] -= (learning_rate * m_hat) / (std::sqrt(v_hat) + epsilon);
        }
        
        // Update bias
        // Update biased first moment estimate
        bias_m[i] = beta1 * bias_m[i] + (1.0 - beta1) * bias_gradients[i];
        
        // Update biased second moment estimate
        bias_v[i] = beta2 * bias_v[i] 
                   + (1.0 - beta2) * bias_gradients[i] * bias_gradients[i];
        
        // Compute bias-corrected moments
        double m_hat = bias_m[i] / bias_correction1;
        double v_hat = bias_v[i] / bias_correction2;
        
        biases[i] -= (learning_rate * m_hat) / (std::sqrt(v_hat) + epsilon);
    }
}

void ADAM::reset() {

    timestep = 0;
    
    // Clear all moment estimates
    for (auto& row : weight_m) {
        std::fill(row.begin(), row.end(), 0.0);
    }
    std::fill(bias_m.begin(), bias_m.end(), 0.0);
    
    for (auto& row : weight_v) {
        std::fill(row.begin(), row.end(), 0.0);
    }
    std::fill(bias_v.begin(), bias_v.end(), 0.0);
}