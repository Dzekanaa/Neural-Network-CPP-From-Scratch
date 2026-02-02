#pragma once
#include <vector>
#include "Neuron.hpp"

/**
 * @brief A class representing a layer of neurons in a neural network.
 * @author Dzekanaa
 */
class Layer
{
private:
    std::vector<Neuron> neurons;
    int numNeurons;
    int numInputsPerNeuron;

public:
    Layer(int numNeurons, int numInputsPerNeuron);
    ~Layer() {}

    /**
     * @brief Performs a forward pass through the layer.
     * @param inputs A vector of input values to the layer.
     * @return A vector containing the outputs of the layer's neurons.
     */
    std::vector<double> forward(const std::vector<double> &inputs);

    /**
     * @brief Retrieves the outputs of all neurons in the layer.
     * @return A vector containing the outputs of the neurons.
     */
    std::vector<double> getOutputs() const;

    // Getters
    std::vector<Neuron> &getNeurons() { return neurons; }
    const std::vector<Neuron> &getNeurons() const { return neurons; }
    int getNumNeurons() const { return numNeurons; }
};