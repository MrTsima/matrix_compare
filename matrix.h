#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <stdexcept>
#include <functional>

class Matrix {
public:
    Matrix() : rows_(0), cols_(0) {}

    Matrix(int rows, int cols);

    int getRows() const { return rows_; }

    int getCols() const { return cols_; }

    int &operator()(int i, int j);

    const int &operator()(int i, int j) const;

    bool operator==(const Matrix &other) const;

    Matrix operator+(const Matrix &other) const;

    Matrix operator-(const Matrix &other) const;

    Matrix multiply(const Matrix &other) const;

    Matrix strassen_multiply(const Matrix &other) const;

private:
    int rows_;
    int cols_;
    std::vector<int> data_;
};

Matrix standard_multiply(const Matrix &A, const Matrix &B);

Matrix strassen_multiply(const Matrix &A, const Matrix &B);

Matrix multithreaded_standard_multiply(const Matrix &A, const Matrix &B, int num_threads);

Matrix multithreaded_strassen_multiply(const Matrix &A, const Matrix &B, int num_threads);

#endif // MATRIX_H