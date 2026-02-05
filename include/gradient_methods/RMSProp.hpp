#pragma once
#include "Optimizer.hpp"

/**
 * @brief RMSProp (Root Mean Square Propagation) Optimizer
 * 
 * Update rule:
 *   cache_w = decay * cache_w + (1 - decay) * (dL/dw)^2
 *   w_new = w_old - (learning_rate / sqrt(cache_w + epsilon)) * dL/dw
 * 
 * decay is the decay rate (typically 0.9 or 0.99)
 * 'epsilon' is a small constant for preventing division by zero
 * 
 * @author Jokula1
 */
class RMSProp : public Optimizer {
private:
    std::vector<std::vector<double>> weight_cache;  // Moving average of squared gradients for weights
    std::vector<double> bias_cache;                 // Moving average of squared gradients for biases
    double decay_rate;                              // Decay rate for moving average (greek ro)
    double epsilon;                                 // Small constant to prevent division by zero
    bool initialized;                               // Whether cache has been initialized
    
public:
    /**
     * @brief Constructor
     * @param lr Learning rate (typical values: 0.001 to 0.01)
     * @param decay Decay rate for moving average (typical values: 0.9 or 0.99)
     * @param eps Epsilon value for numerical stability (default: 1e-8)
     */
    explicit RMSProp(double lr = 0.001, double decay = 0.9, double eps = 1e-8);
    
    ~RMSProp() override = default;
    
    /**
     * @brief Update weights and biases using RMSProp algorithm
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
    
    void set_decay_rate(double decay);
    double get_decay_rate() const { return decay_rate; }
    
private:
    /**
     * @brief Initialize cache arrays to match weight/bias dimensions
     */
    void initialize_cache(size_t num_neurons, size_t num_inputs);
};