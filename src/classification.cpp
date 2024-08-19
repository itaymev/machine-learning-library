#include "classification.hpp"
#include <cmath>
#include <stdexcept>

namespace ml_library {

LogisticRegression::LogisticRegression(double learning_rate, int max_iters)
    : learning_rate_(learning_rate), max_iters_(max_iters), bias_(0) {}

void LogisticRegression::fit(const Matrix& X, const Vector& y) {
    int rows = X.rows();
    int cols = X.cols();
    weights_ = Vector(cols);

    for (int iter = 0; iter < max_iters_; ++iter) {
        Vector predictions(rows);
        for (int i = 0; i < rows; ++i) {
            double linear_model = bias_;
            for (int j = 0; j < cols; ++j) {
                linear_model += weights_[j] * X[i][j];
            }
            predictions[i] = sigmoid(linear_model);
        }

        Vector errors(rows);
        for (int i = 0; i < rows; ++i) {
            errors[i] = y[i] - predictions[i];
        }

        for (int j = 0; j < cols; ++j) {
            double gradient = 0;
            for (int i = 0; i < rows; ++i) {
                gradient += errors[i] * X[i][j];
            }
            weights_[j] += learning_rate_ * gradient / rows;
        }

        double bias_gradient = 0;
        for (int i = 0; i < rows; ++i) {
            bias_gradient += errors[i];
        }
        bias_ += learning_rate_ * bias_gradient / rows;
    }
}

Vector LogisticRegression::predict(const Matrix& X) const {
    int rows = X.rows();
    Vector predictions(rows);
    for (int i = 0; i < rows; ++i) {
        double linear_model = bias_;
        for (int j = 0; j < X.cols(); ++j) {
            linear_model += weights_[j] * X[i][j];
        }
        predictions[i] = sigmoid(linear_model) >= 0.5 ? 1.0 : 0.0;
    }
    return predictions;
}

double LogisticRegression::sigmoid(double z) const {
    return 1.0 / (1.0 + std::exp(-z));
}

} // namespace ml_library
