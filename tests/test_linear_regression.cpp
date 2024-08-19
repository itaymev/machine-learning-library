#include "linear_regression.hpp"
#include <iostream>
#include <cassert>

void test_linear_regression() {
    ml_library::Matrix X = {{1, 2}, {2, 3}, {3, 4}, {4, 5}};
    ml_library::Vector y = {3, 5, 7, 9};

    ml_library::LinearRegression lr;
    lr.fit(X, y);
    ml_library::Vector predictions = lr.predict(X);

    for (int i = 0; i < y.size(); ++i) {
        assert(std::abs(predictions[i] - y[i]) < 1e-6);
    }

    std::cout << "Linear Regression test passed!" << std::endl;
}

int main() {
    test_linear_regression();
    return 0;
}
