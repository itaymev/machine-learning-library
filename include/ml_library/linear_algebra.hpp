#ifndef LINEAR_ALGEBRA_HPP
#define LINEAR_ALGEBRA_HPP

#include <vector>

namespace ml_library {

class Vector;

class Matrix {
public:
    Matrix(); // Default constructor
    Matrix(int rows, int cols);
    Matrix(const std::vector<std::vector<double>>& values);
    Matrix transpose() const; 
    Matrix inverse() const; 
    Matrix operator*(const Matrix& other) const; 
    Vector operator*(const Vector& vec) const;
    std::vector<double>& operator[](int index);
    const std::vector<double>& operator[](int index) const;
    int rows() const;
    int cols() const;
    void push_back(const std::vector<double>& row);

private:
    std::vector<std::vector<double>> values_;
};

class Vector {
public:
    Vector(); // Default constructor
    Vector(int size);
    Vector(const std::vector<double>& values);
    double dot(const Vector& other) const;
    Vector operator+(const Vector& other) const;
    Vector operator-(const Vector& other) const;
    double& operator[](int index);
    const double& operator[](int index) const;
    int size() const;
    void push_back(double value);

    std::vector<double>::iterator begin();
    std::vector<double>::iterator end();
    std::vector<double>::const_iterator begin() const;
    std::vector<double>::const_iterator end() const;

private:
    std::vector<double> values_;
};

} // namespace ml_library

#endif // LINEAR_ALGEBRA_HPP