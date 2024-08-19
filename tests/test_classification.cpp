#include "classification.hpp"
#include <iostream>
#include <cassert>

void test_logistic_regression() {
    ml_library::Matrix X = {{1, 2}, {2, 3}, {3, 4}, {4, 5}};
    ml_library::Vector y = {0, 0, 1, 1};

    ml_library::LogisticRegression lr;
    lr.fit(X, y);
    ml_library::Vector predictions = lr.predict(X);

    for (int i = 0; i < y.size(); ++i) {
        assert(predictions[i] == y[i]);
    }

    std::cout << "Logistic Regression test passed!" << std::endl;
}

int main() {
    test_logistic_regression();
    return 0;
}
