#include "linear_regression.hpp"
#include <stdexcept>
#include <cmath>

namespace ml_library {

LinearRegression::LinearRegression() : bias_(0) {}

void LinearRegression::fit(const Matrix& X, const Vector& y) {
    int rows = X.rows();
    int cols = X.cols();

    // Add a column of ones to X for the bias term
    Matrix X_b(rows, cols + 1);
    for (int i = 0; i < rows; ++i) {
        X_b[i][0] = 1.0;
        for (int j = 0; j < cols; ++j) {
            X_b[i][j + 1] = X[i][j];
        }
    }

    // Normal Equation: (X^T * X)^-1 * X^T * y
    Matrix X_b_T = X_b.transpose();
    Matrix X_b_T_X_b = X_b_T * X_b;
    Matrix X_b_T_X_b_inv = X_b_T_X_b.inverse();
    Vector X_b_T_y = X_b_T * y;

    Vector theta = X_b_T_X_b_inv * X_b_T_y;

    bias_ = theta[0];
    weights_ = Vector(std::vector<double>(theta.begin() + 1, theta.end()));
}

Vector LinearRegression::predict(const Matrix& X) const {
    int rows = X.rows();
    Vector predictions(rows);
    for (int i = 0; i < rows; ++i) {
        predictions[i] = bias_;
        for (int j = 0; j < X.cols(); ++j) {
            predictions[i] += weights_[j] * X[i][j];
        }
    }
    return predictions;
}

} // namespace ml_library