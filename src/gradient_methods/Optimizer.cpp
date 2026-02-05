#include "../include/gradient_methods/Optimizer.hpp"

Optimizer::Optimizer(double lr) : learning_rate(lr) {}

void Optimizer::set_learning_rate(double lr) {
    learning_rate = lr;
}

double Optimizer::get_learning_rate() const {
    return learning_rate;
}

void Optimizer::reset() {
    // Default implementation does nothing
    // Derived classes can override if they maintain state
}