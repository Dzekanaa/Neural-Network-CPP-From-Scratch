#pragma once
#include <cstddef>
#include "../gradient_methods/Optimizer.hpp"

/**
 * @brief ADAM (Adaptive Moment Estimation) Optimizer
 * 
 * Update rule:
 *   m_w = beta1 * m_w + (1 - beta1) * dL/dw         (first moment - momentum)
 *   v_w = beta2 * v_w + (1 - beta2) * (dL/dw)^2      (second moment - squared gradient)
 *   
 *   m_hat = m_w / (1 - beta1^t)                  (bias correction)
 *   v_hat = v_w / (1 - beta2^t)                  (bias correction)
 *   
 *   w_new = w_old - (learning_rate * m_hat) / (sqrt(v_hat) + epsilon)
 * 
 *   beta1 is the decay rate for first moment (typically 0.9)
 *   beta2 is the decay rate for second moment (typically 0.999)
 *   t is the timestep (iteration count)
 *   epislon is a small constant to prevent deviding by zero
 *
 * @author Jokula1
 */
class ADAM : public Optimizer {
private:
    // First moment (momentum)
    std::vector<std::vector<double>> weight_m;  // First moment for weights
    std::vector<double> bias_m;                 // First moment for biases
    
    // Second moment (RMSProp-like)
    std::vector<std::vector<double>> weight_v;  // Second moment for weights
    std::vector<double> bias_v;                 // Second moment for biases
    
    double beta1;                               // Decay rate for first moment
    double beta2;                               // Decay rate for second moment
    double epsilon;                             // Small constant to prevent division by zero
    int timestep;                               // Current timestep (for bias correction)
    bool initialized;                           // Whether moments have been initialized
    
public:
    /**
     * @brief Constructor
     * @param lr Learning rate [0.001, 0.01]
     * @param b1 Beta1 - decay rate for first moment
     * @param b2 Beta2 - decay rate for second moment
     * @param eps Epsilon value for numerical stability
     */
    explicit ADAM(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8);
    
    ~ADAM() override = default;
    
    /**
     * @brief Update weights and biases using ADAM algorithm
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
    
    void set_beta1(double b1) { beta1 = b1; }
    void set_beta2(double b2) { beta2 = b2; }
    double get_beta1() const { return beta1; }
    double get_beta2() const { return beta2; }
    
    int get_timestep() const { return timestep; }
    
private:
    void initialize_moments(size_t num_neurons, size_t num_inputs);
};