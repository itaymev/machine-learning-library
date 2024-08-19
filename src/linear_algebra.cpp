#include "linear_algebra.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm> // for std::swap

namespace ml_library {

// Matrix class implementation
Matrix::Matrix() : values_(0, std::vector<double>(0)) {} // Default constructor

Matrix::Matrix(int rows, int cols) : values_(rows, std::vector<double>(cols)) {}

Matrix::Matrix(const std::vector<std::vector<double>>& values) : values_(values) {}

Matrix Matrix::transpose() const {
    int rows = this->rows();
    int cols = this->cols();
    Matrix result(cols, rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[j][i] = values_[i][j];
        }
    }
    return result;
}

Matrix Matrix::inverse() const {
    int n = this->rows();
    if (n != this->cols()) {
        throw std::invalid_argument("Matrix must be square to invert");
    }

    Matrix result(n, n);
    // Implement matrix inversion logic here
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (this->cols() != other.rows()) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }
    int rows = this->rows();
    int cols = other.cols();
    int common_dim = this->cols();
    Matrix result(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = 0;
            for (int k = 0; k < common_dim; ++k) {
                result[i][j] += values_[i][k] * other[k][j];
            }
        }
    }
    return result;
}

Vector Matrix::operator*(const Vector& vec) const {
    if (this->cols() != vec.size()) {
        throw std::invalid_argument("Matrix and vector dimensions do not match for multiplication");
    }
    int rows = this->rows();
    Vector result(rows);
    for (int i = 0; i < rows; ++i) {
        result[i] = 0;
        for (int j = 0; j < this->cols(); ++j) {
            result[i] += values_[i][j] * vec[j];
        }
    }
    return result;
}

std::vector<double>& Matrix::operator[](int index) {
    if (index < 0 || index >= this->rows()) {
        throw std::out_of_range("Matrix index out of range");
    }
    return values_[index];
}

const std::vector<double>& Matrix::operator[](int index) const {
    if (index < 0 || index >= this->rows()) {
        throw std::out_of_range("Matrix index out of range");
    }
    return values_[index];
}

int Matrix::rows() const {
    return values_.size();
}

int Matrix::cols() const {
    return values_.empty() ? 0 : values_[0].size();
}

void Matrix::push_back(const std::vector<double>& row) {
    if (!values_.empty() && row.size() != values_[0].size()) {
        throw std::invalid_argument("Row size does not match matrix column size");
    }
    values_.push_back(row);
}

// Vector class implementation
Vector::Vector() : values_(0) {} // Default constructor

Vector::Vector(int size) : values_(size) {}

Vector::Vector(const std::vector<double>& values) : values_(values) {}

double Vector::dot(const Vector& other) const {
    if (this->size() != other.size()) {
        throw std::invalid_argument("Vector sizes do not match for dot product");
    }
    double result = 0;
    for (int i = 0; i < this->size(); ++i) {
        result += values_[i] * other[i];
    }
    return result;
}

Vector Vector::operator+(const Vector& other) const {
    if (this->size() != other.size()) {
        throw std::invalid_argument("Vector sizes do not match for addition");
    }
    Vector result(this->size());
    for (int i = 0; i < this->size(); ++i) {
        result[i] = values_[i] + other[i];
    }
    return result;
}

Vector Vector::operator-(const Vector& other) const {
    if (this->size() != other.size()) {
        throw std::invalid_argument("Vector sizes do not match for subtraction");
    }
    Vector result(this->size());
    for (int i = 0; i < this->size(); ++i) {
        result[i] = values_[i] - other[i];
    }
    return result;
}

double& Vector::operator[](int index) {
    if (index < 0 || index >= this->size()) {
        throw std::out_of_range("Vector index out of range");
    }
    return values_[index];
}

const double& Vector::operator[](int index) const {
    if (index < 0 || index >= this->size()) {
        throw std::out_of_range("Vector index out of range");
    }
    return values_[index];
}

int Vector::size() const {
    return values_.size();
}

void Vector::push_back(double value) {
    values_.push_back(value);
}

std::vector<double>::iterator Vector::begin() {
    return values_.begin();
}

std::vector<double>::iterator Vector::end() {
    return values_.end();
}

std::vector<double>::const_iterator Vector::begin() const {
    return values_.begin();
}

std::vector<double>::const_iterator Vector::end() const {
    return values_.end();
}

Vector operator*(const Matrix& mat, const Vector& vec) {
    return mat.operator*(vec);
}

} // namespace ml_library