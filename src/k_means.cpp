#include "k_means.hpp"
#include <limits>
#include <cmath>
#include <stdexcept>

namespace ml_library {

KMeans::KMeans(int k, int max_iters) : k_(k), max_iters_(max_iters) {}

void KMeans::fit(const Matrix& X) {
    int rows = X.rows();
    int cols = X.cols();

    // Initialize centroids randomly
    centroids_ = Matrix(k_, cols);
    for (int i = 0; i < k_; ++i) {
        for (int j = 0; j < cols; ++j) {
            centroids_[i][j] = X[i][j];
        }
    }

    for (int iter = 0; iter < max_iters_; ++iter) {
        // Assign clusters
        std::vector<std::vector<int>> clusters(k_);
        for (int i = 0; i < rows; ++i) {
            double min_dist = std::numeric_limits<double>::max();
            int cluster_idx = 0;
            for (int j = 0; j < k_; ++j) {
                double dist = distance(X[i], centroids_[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    cluster_idx = j;
                }
            }
            clusters[cluster_idx].push_back(i);
        }

        // Update centroids
        for (int j = 0; j < k_; ++j) {
            Matrix cluster_points(clusters[j].size(), cols);
            for (int i = 0; i < clusters[j].size(); ++i) {
                cluster_points[i] = X[clusters[j][i]];
            }
            centroids_[j] = compute_centroid(cluster_points);
        }
    }
}

Vector KMeans::predict(const Matrix& X) const {
    int rows = X.rows();
    Vector labels(rows);
    for (int i = 0; i < rows; ++i) {
        double min_dist = std::numeric_limits<double>::max();
        int cluster_idx = 0;
        for (int j = 0; j < k_; ++j) {
            double dist = distance(X[i], centroids_[j]);
            if (dist < min_dist) {
                min_dist = dist;
                cluster_idx = j;
            }
        }
        labels[i] = cluster_idx;
    }
    return labels;
}

Matrix KMeans::get_centroids() const {
    return centroids_;
}

double KMeans::distance(const Vector& a, const Vector& b) const {
    double sum = 0;
    for (int i = 0; i < a.size(); ++i) {
        sum += std::pow(a[i] - b[i], 2);
    }
    return std::sqrt(sum);
}

Vector KMeans::compute_centroid(const Matrix& points) const {
    int rows = points.rows();
    int cols = points.cols();
    Vector centroid(cols);
    for (int j = 0; j < cols; ++j) {
        double sum = 0;
        for (int i = 0; i < rows; ++i) {
            sum += points[i][j];
        }
        centroid[j] = sum / rows;
    }
    return centroid;
}

} // namespace ml_library
