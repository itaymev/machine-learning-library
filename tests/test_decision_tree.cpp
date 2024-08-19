#include "decision_tree.hpp"
#include <iostream>
#include <cassert>

void test_decision_tree() {
    ml_library::Matrix X = {{1, 2}, {2, 3}, {3, 4}, {4, 5}};
    ml_library::Vector y = {0, 0, 1, 1};

    ml_library::DecisionTree dt;
    dt.fit(X, y);
    ml_library::Vector predictions = dt.predict(X);

    for (int i = 0; i < y.size(); ++i) {
        assert(predictions[i] == y[i]);
    }

    std::cout << "Decision Tree test passed!" << std::endl;
}

int main() {
    test_decision_tree();
    return 0;
}
