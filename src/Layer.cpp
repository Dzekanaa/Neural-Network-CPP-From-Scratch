#include "Layer.hpp"

Layer::Layer(int numNeurons, int numInputsPerNeuron)
{
    neurons.reserve(numNeurons);
    for (int i = 0; i < numNeurons; ++i)
    {
        neurons.emplace_back(numInputsPerNeuron);
    }
}

std::vector<double> Layer::forward(const std::vector<double> &inputs, const ActivationFunc &func)
{
    std::vector<double> outputs;
    outputs.reserve(neurons.size());

    for (Neuron &neuron : neurons)
    {
        outputs.push_back(neuron.activate(inputs, func));
    }
    return outputs;
}

std::vector<double> Layer::getOutputs() const
{
    std::vector<double> outputs;
    outputs.reserve(neurons.size());
    for (const Neuron &neuron : neurons)
    {
        outputs.push_back(neuron.getOutput());
    }
    return outputs;
}
