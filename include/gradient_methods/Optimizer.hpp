#pragma once

#include <vector>

/**
 * Abstract base class for all optimization algorithms
 * 
 * This class defines the interface that all optimizers must implement.
 * Optimizers are responsible for updating network weights and biases
 * based on computed gradients during backpropagation.
 * 
 * Derived classes implement specific optimization strategies:
 * - Steepest Descent (basic gradient descent)
 * - AdaGrad (adaptive learning rates)
 * - Adam (adaptive moment estimation)
 * - RMSProp (root mean square propagation)
 * @author Jokula1
 */
class Optimizer {
protected:
    double learning_rate;
    
public:
    /**
     * Constructor
     * @param lr Learning rate for the optimizer
     */
    explicit Optimizer(double lr);
    
    virtual ~Optimizer() = default;
    
    /**
     * @brief Update weights and biases using the computed gradients
     * 
     * @param weights Network weights - MODIFIED (pass by reference)
     * @param biases Network biases - MODIFIED (pass by reference)
     * @param weight_gradients Computed gradients for weights (pass by const reference)
     * @param bias_gradients Computed gradients for biases (pass by const reference)
     */
    virtual void update(std::vector<std::vector<double>>& weights,
                       std::vector<double>& biases,
                       const std::vector<std::vector<double>>& weight_gradients,
                       const std::vector<double>& bias_gradients) = 0;
    
    /**
     * Set a new learning rate
     * @param lr New learning rate value
     */
    virtual void set_learning_rate(double lr);
    
    /**
     * Get current learning rate
     * @return Current learning rate value
     */
    double get_learning_rate() const;
    
    /**
     * Reset optimizer state (useful when starting new training)
     * Default implementation does nothing - override if optimizer maintains state
     */
    virtual void reset();
};