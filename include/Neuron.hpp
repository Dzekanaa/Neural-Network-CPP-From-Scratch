#pragma once
#include <vector>
#include "ActivationFunctions.hpp"

/**
 * @brief A class representing a single neuron in a neural network with weights, bias, and output.
 * @author Dzekanaa
 */
class Neuron
{
private:
    std::vector<double> weights;
    double bias;
    double output;
    double delta; // For storing the delta value during backpropagation

public:
    /**
     * @brief Constructs a Neuron with a specified number of weights.
     * @param numWeights The number of weights for the neuron.
     */
    Neuron(int numWeights);
    ~Neuron() {}

    /**
     * @brief Activates the neuron using the provided inputs and activation function.
     * @param inputs A vector of input values to the neuron.
     * @param activationFunction The activation function to apply.
     * @return The output of the neuron after activation.
     */
    double activate(const std::vector<double> &inputs, const ActivationFunc &activationFunction);

    /**
     * @brief Initializes the weights of the neuron with random values.
     */
    void initializeWeights(int numWeights);

    double getOutput() const { return output; }
    void setOutput(double val) { output = val; }
    const std::vector<double> &getWeights() const { return weights; }
    void setWeights(const std::vector<double> &vals) { weights = vals; }
    double getBias() const { return bias; }
    void setBias(double val) { bias = val; }
    double getDelta() const { return delta; }
    void setDelta(double val) { delta = val; }
};