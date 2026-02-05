#include "Network.hpp"
#include <iostream>

Network::Network(const std::vector<int> &layerSizes, const Activation &activation, double learningRate)
    : layerSizes(layerSizes), learningRate(learningRate), activation(activation)
{

    for (size_t i = 1; i < layerSizes.size(); ++i)
    {
        layers.emplace_back(layerSizes[i], layerSizes[i - 1]);
    }
}

void Network::setActivation(const Activation &newActivation)
{
    activation = newActivation;
}

std::vector<double> Network::forward(const std::vector<double> &inputs)
{
    std::vector<double> currentInputs = inputs;

    for (auto &layer : layers)
    {
        currentInputs = layer.forward(currentInputs, activation.func);
    }

    return currentInputs;
}

void Network::backward(const std::vector<double>& expectedOutputs)
{
    // Output layer deltas
    Layer& outputLayer = layers.back();

    for (size_t i = 0; i < outputLayer.getNeurons().size(); ++i)
    {
        Neuron& neuron = outputLayer.getNeurons()[i];
        double output = neuron.getOutput();
        double error = targets[i] - output;
        neuron.setDelta(error * activation.derivative(output));
    }

    // Hidden layer deltas (backpropagate)
    for (int l = layers.size() - 2; l >= 0; --l)
    {
        Layer& currentLayer = layers[l];
        Layer& nextLayer = layers[l + 1];

        for (size_t i = 0; i < currentLayer.getNeurons().size(); ++i)
        {
            Neuron& neuron = currentLayer.getNeurons()[i];
            double error = 0.0;

            for (auto& nextNeuron : nextLayer.getNeurons())
            {
                error += nextNeuron.getWeights()[i] * nextNeuron.getDelta();
            }

            neuron.setDelta(error * activation.derivative(neuron.getOutput()));
        }
    }

    // Update weights and biases
    std::vector<double> prevOutputs = inputs;

    for (size_t l = 0; l < layers.size(); ++l)
    {
        Layer& layer = layers[l];

        for (auto& neuron : layer.getNeurons())
        {
            std::vector<double> weights = neuron.getWeights();

            for (size_t w = 0; w < weights.size(); ++w)
            {
                weights[w] += learningRate * neuron.getDelta() * prevOutputs[w];
            }

            neuron.setWeights(weights)
                neuron.setBias(neuron.getBias() + learningRate * neuron.getDelta());
        }

        prevOutputs = layer.getOutputs();
    }
}

void Network::train(const std::vector<double> &inputs, const std::vector<double> &targets)
{
    forward(inputs);
    backwards(inputs, targets);
}

void Network::fit(const std::vector<std::vector<double>> &inputData,
                  const std::vector<std::vector<double>> &targetData,
                  int epochs,
                  bool verbose)
{
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        for (size_t i = 0; i < inputData.size(); ++i)
        {
            train(inputData[i], targetData[i]);
        }

        if (verbose && epoch % 1000 == 0)
        {
            std::cout << "Epoch " << epoch << ", MSE: " << calculateLoss(inputData, targetData) << std::endl;
        }
    }
}

std::vector<double> Network::predict(const std::vector<double> &inputs)
{
    return forward(inputs);
}

double Network::calculateLoss(const std::vector<std::vector<double>> &inputData,
                              const std::vector<std::vector<double>> &targetData)
{
    double totalError = 0.0;

    for (size_t i = 0; i < inputData.size(); ++i)
    {
        std::vector<double> output = predict(inputData[i]);
        for (size_t j = 0; j < output.size(); ++j)
        {
            double error = targetData[i][j] - output[j];
            totalError += error * error;
        }
    }

    return totalError / inputData.size();
}