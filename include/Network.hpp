#pragma once
#include <vector>
#include "Layer.hpp"

/**
 * @brief A class representing a neural network composed of multiple layers.
 * @author Dzekanaa
 */
class Network
{
private:
    std::vector<Layer> layers;
    std::vector<int> layerSizes; // Number of neurons in each layer
    double learningRate;

public:
    /**
     * @brief Constructs a Network with specified layer sizes and learning rate.
     * @param layerSizes A vector containing the number of neurons in each layer.
     * @param learningRate The learning rate for training the network.
     *
     * @example
     * // layerSizes = {3, 5, 2} creates a network with:
     * // - Input layer with 3 neurons
     * // - Hidden layer with 5 neurons
     * // - Output layer with 2 neurons
     */
    Network(const std::vector<int> &layerSizes, double learningRate = 0.1);
    ~Network() {}

    /**
     * @brief Performs a forward pass through the network.
     * @param inputs A vector of input values to the network.
     * @return A vector containing the outputs of the network's final layer.
     */
    std::vector<double> forward(const std::vector<double> &inputs);

    /**
     * @brief Performs a backward pass through the network to update weights.
     * @param expectedOutputs A vector of expected output values for training.
     */
    void backward(const std::vector<double> &expectedOutputs);

    /**
     * @brief Trains the network on a single input-output pair.
     * @param trainingInput A vector of input values for training.
     * @param trainingOutput A vector of expected output values for training.
     */
    void train(const std::vector<double> &trainingInput,
               const std::vector<double> &trainingOutput);

    /**
     * @brief Trains the network over multiple epochs using the provided training data.
     * @param trainingInputs A vector of input vectors for training.
     * @param trainingOutputs A vector of expected output vectors for training.
     * @param epochs The number of training epochs.
     */
    void fit(const std::vector<std::vector<double>> &trainingInputs,
             const std::vector<std::vector<double>> &trainingOutputs,
             int epochs);

    /**
     * @brief Makes a prediction using the trained network.
     * @param input A vector of input values for prediction.
     * @return A vector containing the predicted output values.
     */
    std::vector<double> predict(const std::vector<double> &input);

    // Getters
    std::vector<Layer> &getLayers() { return layers; }
    const std::vector<int> &getLayerSizes() const { return layerSizes; }
};