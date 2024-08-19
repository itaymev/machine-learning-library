#include "k_means.hpp"
#include <iostream>
#include <cassert>

void test_k_means() {
    ml_library::Matrix X = {{1, 2}, {1, 4}, {1, 0}, {10, 2}, {10, 4}, {10, 0}};
    ml_library::KMeans kmeans(2);
    kmeans.fit(X);
    ml_library::Vector labels = kmeans.predict(X);

    // Check if the labels are consistent with the clusters
    assert(labels[0] == labels[1]);
    assert(labels[1] == labels[2]);
    assert(labels[3] == labels[4]);
    assert(labels[4] == labels[5]);
    assert(labels[0] != labels[3]);

    std::cout << "K-Means test passed!" << std::endl;
}

int main() {
    test_k_means();
    return 0;
}
