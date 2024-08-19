#ifndef CLASSIFICATION_HPP
#define CLASSIFICATION_HPP

#include "linear_algebra.hpp"

namespace ml_library {

class LogisticRegression {
public:
    LogisticRegression(double learning_rate = 0.01, int max_iters = 1000);
    void fit(const Matrix& X, const Vector& y);
    Vector predict(const Matrix& X) const;

private:
    Vector weights_;
    double bias_;
    double learning_rate_;
    int max_iters_;

    double sigmoid(double z) const;
};

} // namespace ml_library

#endif // CLASSIFICATION_HPP
