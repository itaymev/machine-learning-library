#include "decision_tree.hpp"
#include <limits>
#include <cmath>
#include <stdexcept>

namespace ml_library {

DecisionTree::DecisionTree(int max_depth) : max_depth_(max_depth) {}

void DecisionTree::fit(const Matrix& X, const Vector& y) {
    root_ = build_tree(X, y, 0);
}

Vector DecisionTree::predict(const Matrix& X) const {
    int rows = X.rows();
    Vector predictions(rows);
    for (int i = 0; i < rows; ++i) {
        predictions[i] = predict(X[i], root_);
    }
    return predictions;
}

std::shared_ptr<DecisionTree::Node> DecisionTree::build_tree(const Matrix& X, const Vector& y, int depth) {
    if (depth >= max_depth_ || y.size() <= 1) {
        return std::make_shared<Node>(Node{0, 0, calculate_leaf_value(y), nullptr, nullptr, true});
    }

    int best_feature = 0;
    double best_threshold = 0;
    double best_gini = std::numeric_limits<double>::max();
    int rows = X.rows();
    int cols = X.cols();

    for (int i = 0; i < cols; ++i) {
        for (int j = 0; j < rows; ++j) {
            double threshold = X[j][i];
            double gini = calculate_gini(X, y, i, threshold);
            if (gini < best_gini) {
                best_gini = gini;
                best_feature = i;
                best_threshold = threshold;
            }
        }
    }

    Matrix left_X, right_X;
    Vector left_y, right_y;
    for (int i = 0; i < rows; ++i) {
        if (X[i][best_feature] <= best_threshold) {
            left_X.push_back(X[i]);
            left_y.push_back(y[i]);
        } else {
            right_X.push_back(X[i]);
            right_y.push_back(y[i]);
        }
    }

    auto left_node = build_tree(left_X, left_y, depth + 1);
    auto right_node = build_tree(right_X, right_y, depth + 1);

    return std::make_shared<Node>(Node{best_feature, best_threshold, 0, left_node, right_node, false});
}

double DecisionTree::calculate_gini(const Matrix& X, const Vector& y, int feature_index, double threshold) const {
    int left_count = 0;
    int right_count = 0;
    double left_sum = 0;
    double right_sum = 0;

    for (int i = 0; i < y.size(); ++i) {
        if (X[i][feature_index] <= threshold) {
            left_count++;
            left_sum += y[i];
        } else {
            right_count++;
            right_sum += y[i];
        }
    }

    double left_gini = 1 - std::pow(left_sum / left_count, 2) - std::pow(1 - left_sum / left_count, 2);
    double right_gini = 1 - std::pow(right_sum / right_count, 2) - std::pow(1 - right_sum / right_count, 2);

    return (left_count * left_gini + right_count * right_gini) / y.size();
}

double DecisionTree::calculate_leaf_value(const Vector& y) const {
    double sum = 0;
    for (int i = 0; i < y.size(); ++i) {
        sum += y[i];
    }
    return sum / y.size();
}

double DecisionTree::predict(const Vector& x, std::shared_ptr<Node> node) const {
    if (node->is_leaf) {
        return node->value;
    }
    if (x[node->feature_index] <= node->threshold) {
        return predict(x, node->left);
    } else {
        return predict(x, node->right);
    }
}

} // namespace ml_library
