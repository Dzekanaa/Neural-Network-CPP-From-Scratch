#include "Neuron.hpp"
#include <iostream>
#include <random>

Neuron::Neuron(int numWeights) : output(0.0), delta(0.0)
{
    initializeWeights(numWeights);
}

void Neuron::initializeWeights(int numWeights)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    weights.resize(numWeights);
    for (int i = 0; i < numWeights; ++i)
    {
        weights[i] = dis(gen);
    }
    bias = dis(gen);
}

double Neuron::activate(const std::vector<double> &inputs, const ActivationFunc &func)
{
    if (inputs.size() != weights.size())
    {
        throw std::invalid_argument("Input size does not match weights size.");
    }

    double netInput = bias;
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        netInput += weights[i] * inputs[i];
    }

    output = func(netInput);
    return output;
}