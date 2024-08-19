#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP

#include "linear_algebra.hpp"
#include <vector>
#include <memory>

namespace ml_library {

class DecisionTree {
public:
    DecisionTree(int max_depth = 10);
    void fit(const Matrix& X, const Vector& y);
    Vector predict(const Matrix& X) const;

private:
    struct Node {
        int feature_index;
        double threshold;
        double value;
        std::shared_ptr<Node> left;
        std::shared_ptr<Node> right;
        bool is_leaf;
    };

    std::shared_ptr<Node> root_;
    int max_depth_;

    std::shared_ptr<Node> build_tree(const Matrix& X, const Vector& y, int depth);
    double calculate_gini(const Matrix& X, const Vector& y, int feature_index, double threshold) const;
    double calculate_leaf_value(const Vector& y) const;
    double predict(const Vector& x, std::shared_ptr<Node> node) const;
};

} // namespace ml_library

#endif // DECISION_TREE_HPP
