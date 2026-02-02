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

    Activation activation; // Activation function for the network

public:
    /**
     * @brief Constructs a Network with specified layer sizes and learning rate.
     * @param layerSizes A vector containing the number of neurons in each layer.
     * @param activation The activation function to be used in the network.
     * @param learningRate The learning rate for training the network.
     *
     * @example
     * // layerSizes = {3, 5, 2} creates a network with:
     * // - Input layer with 3 neurons
     * // - Hidden layer with 5 neurons
     * // - Output layer with 2 neurons
     */
    Network(const std::vector<int> &layerSizes, const Activation &activation = Activations::sigmoid(), double learningRate = 0.1);
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
     * @param verbose If true, prints progress during training.
     */
    void fit(const std::vector<std::vector<double>> &trainingInputs,
             const std::vector<std::vector<double>> &trainingOutputs,
             int epochs, bool verbose = true);

    /**
     * @brief Makes a prediction using the trained network.
     * @param input A vector of input values for prediction.
     * @return A vector containing the predicted output values.
     */
    std::vector<double> predict(const std::vector<double> &input);

    /**
     * @brief Calculates the loss between predicted and actual outputs using Mean Squared Error.
     * @param predicted A vector of predicted output vectors.
     * @param actual A vector of actual output vectors.
     * @return The calculated loss value.
     */
    double calculateLoss(const std::vector<std::vector<double>> &predicted, const std::vector<std::vector<double>> &actual);

    // Getters
    std::vector<Layer> &getLayers() { return layers; }
    const std::vector<int> &getLayerSizes() const { return layerSizes; }
    void setActivation(const Activation &act) { activation = act; }
    std::string getActivationName() const { return activation.name; }
};