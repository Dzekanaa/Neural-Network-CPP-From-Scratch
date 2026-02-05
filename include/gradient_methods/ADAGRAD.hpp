#pragma once
#include "Optimizer.hpp"

/**
 * @brief ADAGRAD (Adaptive Gradient) Optimizer
 * 
 * 
 * Update rule:
 *   cache_w = cache_w + (dL/dw)^2
 *   w_new = w_old - (learning_rate / sqrt(cache_w + epsilon)) * dL/dw
 * 
 * 'epsilon' is a small constant to prevent division by zero.
 * 
 * @author Jokula1
 */
class ADAGRAD : public Optimizer {
private:
    std::vector<std::vector<double>> weight_cache;  // Accumulated squared gradients for weights
    std::vector<double> bias_cache;                 // Accumulated squared gradients for biases
    double epsilon;                                 // Small constant to prevent division by zero
    bool initialized;                               // Whether cache has been initialized
    
public:
    /**
     * @brief Constructor
     * @param lr Learning rate [0.01, 0.1]
     * @param eps Epsilon value for numerical stability (default: 1e-8)
     */
    explicit ADAGRAD(double lr = 0.01, double eps = 1e-8);
    
    ~ADAGRAD() override = default;
    
    /**
     * @brief Update weights and biases using ADAGRAD algorithm
     * 
     * @param weights Network weights - MODIFIED
     * @param biases Network biases - MODIFIED
     * @param weight_gradients Computed gradients for weights
     * @param bias_gradients Computed gradients for biases
     */
    void update(std::vector<std::vector<double>>& weights,
               std::vector<double>& biases,
               const std::vector<std::vector<double>>& weight_gradients,
               const std::vector<double>& bias_gradients) override;
    
    void reset() override;
    
private:
    /**
     * @brief Initialize cache arrays to match weight/bias dimensions
     */
    void initialize_cache(size_t num_neurons, size_t num_inputs);
};