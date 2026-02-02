#include <vector>

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

public:
    Neuron(int numWeights);

    void initializeWeights();
    ~Neuron() {}

    double getOutput() const { return output; }
    void setOutput(double val) { output = val; }
    const std::vector<double> &getWeights() const { return weights; }
    void setWeights(const std::vector<double> &vals) { weights = vals; }
    double getBias() const { return bias; }
    void setBias(double val) { bias = val; }
};