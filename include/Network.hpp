#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <vector>
#include <memory>

// Core components
#include "Layer.hpp"
#include "ActivationFunctions.hpp"

// Optimizers
#include "gradient_methods/Optimizer.hpp"

//
// Optimizer type selector
//
enum class OptimizerType {
    STEEPEST_DESCENT,
    ADAGRAD,
    RMSPROP,
    ADAM
};

class Network
{
public:
    // ==========================================================
    // Constructor
    // ==========================================================
    Network(const std::vector<int>& layerSizes,
            const Activation& activation,
            OptimizerType optimizerType,
            double learningRate);

    // ==========================================================
    // Configuration
    // ==========================================================
    void setOptimizer(OptimizerType optimizerType, double learningRate);
    void setActivation(const Activation& newActivation);

    // ==========================================================
    // Core API
    // ==========================================================
    std::vector<double> forward(const std::vector<double>& inputs);

    void backward(const std::vector<double>& inputs,
                  const std::vector<double>& targets);

    void train(const std::vector<double>& inputs,
               const std::vector<double>& targets);

    void fit(const std::vector<std::vector<double>>& inputData,
             const std::vector<std::vector<double>>& targetData,
             int epochs,
             bool verbose = false);

    std::vector<double> predict(const std::vector<double>& inputs);

    double calculateLoss(const std::vector<std::vector<double>>& inputData,
                         const std::vector<std::vector<double>>& targetData);

private:
    // ==========================================================
    // Network structure
    // ==========================================================
    std::vector<Layer> layers;
    std::vector<int> layerSizes;

    Activation activation;

    // ==========================================================
    // Optimizer
    // ==========================================================
    std::unique_ptr<Optimizer> optimizer;

    // ==========================================================
    // Gradient storage
    // weight_gradients[layer][neuron][weight]
    // bias_gradients[layer][neuron]
    // ==========================================================
    std::vector<std::vector<std::vector<double>>> weight_gradients;
    std::vector<std::vector<double>> bias_gradients;

    // ==========================================================
    // Internal helpers
    // ==========================================================
    void initializeGradients();
    std::unique_ptr<Optimizer> createOptimizer(OptimizerType type, double learningRate);
};

#endif // NETWORK_HPP