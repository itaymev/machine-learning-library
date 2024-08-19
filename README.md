# Machine Learning Library

This library provides implementations of various machine learning algorithms, including Linear Regression, K-Means Clustering, Decision Trees, and Logistic Regression. Each module is designed to be easy to use and integrate into your projects.

## Modules

### Linear Regression

**Header File:** `linear_regression.hpp`  
**Source File:** `linear_regression.cpp`

Linear Regression is a simple and commonly used algorithm for predictive analysis. It models the relationship between a dependent variable and one or more independent variables.

- **Class:** `LinearRegression`
- **Methods:**
  - `LinearRegression()`: Constructor to initialize the model.
  - `void fit(const Matrix& X, const Vector& y)`: Fits the model to the data.
  - `Vector predict(const Matrix& X) const`: Predicts the output for the given input data.

### K-Means Clustering

**Header File:** `k_means.hpp`  
**Source File:** `k_means.cpp`

K-Means is an unsupervised learning algorithm used for clustering. It partitions the data into `k` clusters based on feature similarity.

- **Class:** `KMeans`
- **Methods:**
  - `KMeans(int k, int max_iters = 100)`: Constructor to initialize the model with `k` clusters and a maximum number of iterations.
  - `void fit(const Matrix& X)`: Fits the model to the data.
  - `Vector predict(const Matrix& X) const`: Predicts the cluster for each data point.
  - `Matrix get_centroids() const`: Returns the centroids of the clusters.

### Decision Tree

**Header File:** `decision_tree.hpp`  
**Source File:** `decision_tree.cpp`

Decision Trees are a non-parametric supervised learning method used for classification and regression. They create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

- **Class:** `DecisionTree`
- **Methods:**
  - `DecisionTree(int max_depth = 10)`: Constructor to initialize the model with a maximum depth.
  - `void fit(const Matrix& X, const Vector& y)`: Fits the model to the data.
  - `Vector predict(const Matrix& X) const`: Predicts the output for the given input data.

### Logistic Regression

**Header File:** `classification.hpp`  
**Source File:** `classification.cpp`

Logistic Regression is a statistical method for analyzing a dataset in which there are one or more independent variables that determine an outcome. The outcome is measured with a dichotomous variable (in which there are only two possible outcomes).

- **Class:** `LogisticRegression`
- **Methods:**
  - `LogisticRegression(double learning_rate = 0.01, int max_iters = 1000)`: Constructor to initialize the model with a learning rate and maximum number of iterations.
  - `void fit(const Matrix& X, const Vector& y)`: Fits the model to the data.
  - `Vector predict(const Matrix& X) const`: Predicts the output for the given input data.

### Linear Algebra

**Header File:** `linear_algebra.hpp`  
**Source File:** `linear_algebra.cpp`

This module provides basic linear algebra operations, including matrix and vector operations, which are essential for implementing machine learning algorithms.

- **Classes:**
  - `Matrix`: Represents a matrix and provides methods for matrix operations.
    - **Methods:**
      - `Matrix()`: Default constructor.
      - `Matrix(int rows, int cols)`: Constructor to initialize a matrix with given dimensions.
      - `Matrix(const std::vector<std::vector<double>>& values)`: Constructor to initialize a matrix with given values.
      - `Matrix transpose() const`: Returns the transpose of the matrix.
      - `Matrix inverse() const`: Returns the inverse of the matrix.
      - `Matrix operator*(const Matrix& other) const`: Multiplies two matrices.
      - `Vector operator*(const Vector& vec) const`: Multiplies a matrix with a vector.
      - `std::vector<double>& operator[](int index)`: Accesses a row of the matrix.
      - `const std::vector<double>& operator[](int index) const`: Accesses a row of the matrix (const version).
      - `int rows() const`: Returns the number of rows.
      - `int cols() const`: Returns the number of columns.
      - `void push_back(const std::vector<double>& row)`: Adds a row to the matrix.
  - `Vector`: Represents a vector and provides methods for vector operations.
    - **Methods:**
      - `Vector()`: Default constructor.
      - `Vector(int size)`: Constructor to initialize a vector with a given size.
      - `Vector(const std::vector<double>& values)`: Constructor to initialize a vector with given values.
      - `double dot(const Vector& other) const`: Computes the dot product with another vector.
      - `Vector operator+(const Vector& other) const`: Adds two vectors.
      - `Vector operator-(const Vector& other) const`: Subtracts two vectors.
      - `double& operator[](int index)`: Accesses an element of the vector.
      - `const double& operator[](int index) const`: Accesses an element of the vector (const version).
      - `int size() const`: Returns the size of the vector.
      - `void push_back(double value)`: Add a data element to the vector.
      - `std::vector<double>::iterator begin()`: Returns an iterator to the beginning.
      - `std::vector<double>::iterator end()`: Returns an iterator to the end.
      - `std::vector<double>::const_iterator begin() const`: Returns a const iterator to the beginning.
      - `std::vector<double>::const_iterator end() const`: Returns a const iterator to the end.

## Learning and Sources

The following links and resources were used to implement linear algebra and machine learning methods.

**LibreTexts** - https://math.libretexts.org/Bookshelves/Linear_Algebra
**Data Mining: Concepts and Techniques Third Edition (Jiawei Han, Micheline Kamber and Jian Pei, 2012)** - https://www.sciencedirect.com/book/9780123814791/data-mining-concepts-and-techniques
**Data Mining: Practical Machine Learning Tools and Techniques (Ian H. Witten, Eibe Frank, ... Christopher J. Pal, 2017)** - https://www.sciencedirect.com/book/9780128042915/data-mining

## Usage

To use any of these modules, include the corresponding header file in your project and create an instance of the class. Call the `fit` method to train the model and the `predict` method to make predictions.

Example for Linear Regression:

```cpp
#include "linear_regression.hpp"

int main() {
    ml_library::Matrix X = ...; // Initialize with your data
    ml_library::Vector y = ...; // Initialize with your data

    ml_library::LinearRegression lr;
    lr.fit(X, y);
    ml_library::Vector predictions = lr.predict(X);

    return 0;
}