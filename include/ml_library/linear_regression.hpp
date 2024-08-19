#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include "linear_algebra.hpp"

namespace ml_library {

class LinearRegression {
public:
    LinearRegression();
    void fit(const Matrix& X, const Vector& y);
    Vector predict(const Matrix& X) const;

private:
    Vector weights_;
    double bias_;
};

} // namespace ml_library

#endif // LINEAR_REGRESSION_HPP
