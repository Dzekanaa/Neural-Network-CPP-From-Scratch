#pragma once
#include <cmath>
#include <functional>
#include <string>

/**
 * @brief A collection of activation functions and their derivatives for neural networks.
 */
using ActivationFunc = std::function<double(double)>;

/**
 * @brief A struct representing an activation function and its derivative.
 * @author Dzekanaa
 */
struct Activation
{
    ActivationFunc func;
    ActivationFunc derivative;
    std::string name;
};

namespace Activations
{

    /**
     * @brief Sigmoid activation function and its derivative.
     */
    inline Activation sigmoid()
    {
        return {
            [](double x)
            { return 1.0 / (1.0 + std::exp(-x)); },
            [](double x)
            { return x * (1.0 - x); }, // x is sigmoid output
            "Sigmoid"};
    }

    /**
     * @brief ReLU activation function and its derivative.
     */
    inline Activation relu()
    {
        return {
            [](double x)
            { return std::max(0.0, x); },
            [](double x)
            { return x > 0.0 ? 1.0 : 0.0; },
            "ReLU"};
    }

    /**
     * @brief Leaky ReLU activation function and its derivative.
     */
    inline Activation leakyRelu(double alpha = 0.01)
    {
        return {
            [alpha](double x)
            { return x > 0.0 ? x : alpha * x; },
            [alpha](double x)
            { return x > 0.0 ? 1.0 : alpha; },
            "Leaky ReLU"};
    }

    /**
     * @brief Tanh activation function and its derivative.
     */
    inline Activation tanh()
    {
        return {
            [](double x)
            { return std::tanh(x); },
            [](double x)
            { return 1.0 - x * x; }, // x is tanh output
            "Tanh"};
    }

    /**
     * @brief Linear activation function and its derivative.
     */
    inline Activation linear()
    {
        return {
            [](double x)
            { return x; },
            [](double x)
            { return 1.0; },
            "Linear"};
    }

    /**
     * @brief Softplus activation function and its derivative.
     */
    inline Activation softplus()
    {
        return {
            [](double x)
            { return std::log(1.0 + std::exp(x)); },
            [](double x)
            { return 1.0 / (1.0 + std::exp(-x)); },
            "Softplus"};
    }

    /**
     * @brief ELU activation function and its derivative.
     */
    inline Activation elu(double alpha = 1.0)
    {
        return {
            [alpha](double x)
            { return x > 0.0 ? x : alpha * (std::exp(x) - 1.0); },
            [alpha](double x)
            { return x > 0.0 ? 1.0 : alpha * std::exp(x); },
            "ELU"};
    }

    /**
     * @brief Swish activation function and its derivative.
     */
    inline Activation swish()
    {
        return {
            [](double x)
            {
                double sig = 1.0 / (1.0 + std::exp(-x));
                return x * sig;
            },
            [](double x)
            {
                double sig = 1.0 / (1.0 + std::exp(-x));
                return sig + x * sig * (1.0 - sig);
            },
            "Swish"};
    }

    /**
     * @brief GELU activation function and its derivative.
     */
    inline Activation gelu()
    {
        return {
            [](double x)
            {
                const double c = std::sqrt(2.0 / M_PI);
                return 0.5 * x * (1.0 + std::tanh(c * (x + 0.044715 * x * x * x)));
            },
            [](double x)
            {
                const double c = std::sqrt(2.0 / M_PI);
                double inner = c * (x + 0.044715 * x * x * x);
                double tanhInner = std::tanh(inner);
                double sech2 = 1.0 - tanhInner * tanhInner;
                return 0.5 * (1.0 + tanhInner) + 0.5 * x * sech2 * c * (1.0 + 3.0 * 0.044715 * x * x);
            },
            "GELU"};
    }
}
