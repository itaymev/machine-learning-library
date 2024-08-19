#ifndef K_MEANS_HPP
#define K_MEANS_HPP

#include "linear_algebra.hpp"
#include <vector>

namespace ml_library {

class KMeans {
public:
    KMeans(int k, int max_iters = 100);
    void fit(const Matrix& X);
    Vector predict(const Matrix& X) const;
    Matrix get_centroids() const;

private:
    int k_;
    int max_iters_;
    Matrix centroids_;
    double distance(const Vector& a, const Vector& b) const;
    Vector compute_centroid(const Matrix& points) const;
};

} // namespace ml_library

#endif // K_MEANS_HPP
