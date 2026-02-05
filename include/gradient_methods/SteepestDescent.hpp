#pragma once

#include "Optimizer.hpp"

/**
 * Steepest Descent Optimizer (also known as Gradient Descent)
 * 
 * This is the simplest gradient-based optimization method.
 * It updates parameters by moving in the direction opposite to the gradient.
 * 
 * Update rule:
 *   w_new = w_old - learning_rate * ∂L/∂w
 *   b_new = b_old - learning_rate * ∂L/∂b
 * 
 * Where:
 *   - w are the weights
 *   - b are the biases
 *   - ∂L/∂w and ∂L/∂b are the gradients (partial derivatives of loss)
 *   - learning_rate controls the step size
 * 
 * Advantages:
 *   - Simple and easy to understand
 *   - No additional memory required
 *   - Works well for convex problems
 * 
 * Disadvantages:
 *   - Can be slow to converge
 *   - Sensitive to learning rate choice
 *   - Can get stuck in local minima
 *   - Same learning rate for all parameters
 */
class SteepestDescent : public Optimizer {
public:
    /**
     * Constructor
     * @param lr Learning rate (typical values: 0.001 to 0.1)
     */
    explicit SteepestDescent(double lr = 0.01);
    
    /**
     * Destructor
     */
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