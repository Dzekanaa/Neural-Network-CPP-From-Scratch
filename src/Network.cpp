#include "Network.hpp"

#include "gradient_methods/SteepestDescent.hpp"
#include "gradient_methods/ADAGRAD.hpp"
#include "gradient_methods/RMSProp.hpp"
#include "gradient_methods/ADAM.hpp"

#include <iostream>
#include <stdexcept>

// ==========================================================
// Constructor
// ==========================================================
Network::Network(const std::vector<int>& layerSizes,
                 const Activation& activation,
                 OptimizerType optimizerType,
                 double learningRate)
    : layerSizes(layerSizes), activation(activation)
{
    // Create layers
    for (size_t i = 1; i < layerSizes.size(); ++i)
    {
        layers.emplace_back(layerSizes[i], layerSizes[i - 1]);
    }

    // Initialize gradient storage
    initializeGradients();

    // Create optimizer
    optimizer = createOptimizer(optimizerType, learningRate);
}

// ==========================================================
// Gradient initialization
// ==========================================================
void Network::initializeGradients()
{
    weight_gradients.resize(layers.size());
    bias_gradients.resize(layers.size());

    for (size_t l = 0; l < layers.size(); ++l)
    {
        int numNeurons = layerSizes[l + 1];
        int numInputs  = layerSizes[l];

        weight_gradients[l].resize(
            numNeurons,
            std::vector<double>(numInputs, 0.0)
        );

        bias_gradients[l].resize(numNeurons, 0.0);
    }
}

// ==========================================================
// Optimizer factory
// ==========================================================
std::unique_ptr<Optimizer>
Network::createOptimizer(OptimizerType type, double learningRate)
{
    switch (type)
    {
        case OptimizerType::STEEPEST_DESCENT:
            return std::make_unique<SteepestDescent>(learningRate);

        case OptimizerType::ADAGRAD:
            return std::make_unique<ADAGRAD>(learningRate);

        case OptimizerType::RMSPROP:
            return std::make_unique<RMSProp>(learningRate);

        case OptimizerType::ADAM:
            return std::make_unique<ADAM>(learningRate);

        default:
            throw std::invalid_argument("Unknown optimizer type");
    }
}

// ==========================================================
// Configuration
// ==========================================================
void Network::setOptimizer(OptimizerType optimizerType, double learningRate)
{
    optimizer = createOptimizer(optimizerType, learningRate);
}

void Network::setActivation(const Activation& newActivation)
{
    activation = newActivation;
}

// ==========================================================
// Forward pass
// ==========================================================
std::vector<double>
Network::forward(const std::vector<double>& inputs)
{
    std::vector<double> currentInputs = inputs;

    for (auto& layer : layers)
    {
        currentInputs = layer.forward(currentInputs, activation.func);
    }

    return currentInputs;
}

// ==========================================================
// Backward pass (compute gradients)
// ==========================================================
void Network::backward(const std::vector<double>& inputs,
                       const std::vector<double>& targets)
{
    // ======================================================
    // STEP 1: Output layer deltas
    // ======================================================
    Layer& outputLayer = layers.back();

    for (size_t i = 0; i < outputLayer.getNeurons().size(); ++i)
    {
        Neuron& neuron = outputLayer.getNeurons()[i];

        double output = neuron.getOutput();
        double error  = output - targets[i];

        double delta = error * activation.derivative(output);
        neuron.setDelta(delta);
    }

    // ======================================================
    // STEP 2: Hidden layer deltas (backpropagation)
    // ======================================================
    for (int l = static_cast<int>(layers.size()) - 2; l >= 0; --l)
    {
        Layer& currentLayer = layers[l];
        Layer& nextLayer    = layers[l + 1];

        for (size_t i = 0; i < currentLayer.getNeurons().size(); ++i)
        {
            Neuron& neuron = currentLayer.getNeurons()[i];
            double error  = 0.0;

            for (const auto& nextNeuron : nextLayer.getNeurons())
            {
                error += nextNeuron.getWeights()[i] * nextNeuron.getDelta();
            }

            double delta =
                error * activation.derivative(neuron.getOutput());

            neuron.setDelta(delta);
        }
    }

    // ======================================================
    // STEP 3: Compute gradients
    // ======================================================
    std::vector<double> prevOutputs = inputs;

    for (size_t l = 0; l < layers.size(); ++l)
    {
        Layer& layer = layers[l];

        for (size_t i = 0; i < layer.getNeurons().size(); ++i)
        {
            const Neuron& neuron = layer.getNeurons()[i];
            double delta = neuron.getDelta();

            // Bias gradient
            bias_gradients[l][i] = delta;

            // Weight gradients
            for (size_t j = 0; j < prevOutputs.size(); ++j)
            {
                weight_gradients[l][i][j] = delta * prevOutputs[j];
            }
        }

        prevOutputs = layer.getOutputs();
    }
}

// ==========================================================
// Training step
// ==========================================================
void Network::train(const std::vector<double>& inputs,
                    const std::vector<double>& targets)
{
    // Forward
    forward(inputs);

    // Backward (gradients)
    backward(inputs, targets);

    // Update parameters using optimizer
    for (size_t l = 0; l < layers.size(); ++l)
    {
        Layer& layer = layers[l];

        std::vector<std::vector<double>> weights;
        std::vector<double> biases;

        for (auto& neuron : layer.getNeurons())
        {
            weights.push_back(neuron.getWeights());
            biases.push_back(neuron.getBias());
        }

        optimizer->update(
            weights,
            biases,
            weight_gradients[l],
            bias_gradients[l]
        );

        for (size_t i = 0; i < layer.getNeurons().size(); ++i)
        {
            layer.getNeurons()[i].setWeights(weights[i]);
            layer.getNeurons()[i].setBias(biases[i]);
        }
    }
}

// ==========================================================
// Training loop
// ==========================================================
void Network::fit(const std::vector<std::vector<double>>& inputData,
                  const std::vector<std::vector<double>>& targetData,
                  int epochs,
                  bool verbose)
{
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        for (size_t i = 0; i < inputData.size(); ++i)
        {
            train(inputData[i], targetData[i]);
        }

        if (verbose && epoch % 100 == 0)
        {
            double loss = calculateLoss(inputData, targetData);
            std::cout << "Epoch " << epoch
                      << ", MSE: " << loss << std::endl;
        }
    }
}

// ==========================================================
// Inference
// ==========================================================
std::vector<double>
Network::predict(const std::vector<double>& inputs)
{
    return forward(inputs);
}

// ==========================================================
// Loss (Mean Squared Error)
// ==========================================================
double Network::calculateLoss(
    const std::vector<std::vector<double>>& inputData,
    const std::vector<std::vector<double>>& targetData)
{
    double totalError = 0.0;

    for (size_t i = 0; i < inputData.size(); ++i)
    {
        std::vector<double> output = predict(inputData[i]);

        for (size_t j = 0; j < output.size(); ++j)
        {
            double error = output[j] - targetData[i][j];
            totalError += error * error;
        }
    }

    return totalError / inputData.size();
}