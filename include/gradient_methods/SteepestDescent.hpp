#pragma once

#include "Optimizer.hpp"

/**
 * Steepest Descent Optimizer (also known as Gradient Descent)
 * 
 * This is the simplest gradient-based optimization method.
 * It updates parameters by moving in the direction opposite to the gradient.
 * 
 * Update rule:
 *   w_new = w_old - learning_rate * dL/dw
 *   b_new = b_old - learning_rate * dL/db
 * 
 *   w - weights
 *   b - biases
 *   dL/dw and dL/db    - gradients (partial derivatives of loss)
 *   learning_rate      - controls the step size

 */
class SteepestDescent : public Optimizer {
public:
    /**
     * Constructor
     * @param lr Learning rate [0.001, 0.1]
     */
    explicit SteepestDescent(double lr = 0.01);
    
    ~SteepestDescent() override = default;
    
    /**
     * Update weights and biases using steepest descent
     * 
     * @param weights Network weights - MODIFIED (pass by reference)
     * @param biases Network biases - MODIFIED (pass by reference)
     * @param weight_gradients Computed gradients for weights (pass by const reference)
     * @param bias_gradients Computed gradients for biases (pass by const reference)
     */
    void update(std::vector<std::vector<double>>& weights,
               std::vector<double>& biases,
               const std::vector<std::vector<double>>& weight_gradients,
               const std::vector<double>& bias_gradients) override;
};