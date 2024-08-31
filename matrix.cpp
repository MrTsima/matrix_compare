#include "matrix.h"
#include <stdexcept>
#include <thread>

Matrix::Matrix(int rows, int cols) : rows_(rows), cols_(cols), data_(rows * cols) {}

int &Matrix::operator()(int i, int j) { return data_[i * cols_ + j]; }

const int &Matrix::operator()(int i, int j) const { return data_[i * cols_ + j]; }

bool Matrix::operator==(const Matrix &other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        return false;
    }

    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            if ((*this)(i, j) != other(i, j)) {
                return false;
            }
        }
    }

    return true;
}

Matrix Matrix::operator+(const Matrix &other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions do not match for addition");
    }

    Matrix result(rows_, cols_);
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            result(i, j) = (*this)(i, j) + other(i, j);
        }
    }
    return result;
}

Matrix Matrix::operator-(const Matrix &other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions do not match for subtraction");
    }

    Matrix result(rows_, cols_);
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            result(i, j) = (*this)(i, j) - other(i, j);
        }
    }
    return result;
}

Matrix Matrix::multiply(const Matrix &other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Incompatible matrix dimensions");
    }

    Matrix result(rows_, other.cols_);

    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < other.cols_; ++j) {
            result(i, j) = 0;
            for (int k = 0; k < cols_; ++k) {
                result(i, j) += (*this)(i, k) * other(k, j);
            }
        }
    }

    return result;
}

Matrix Matrix::strassen_multiply(const Matrix &other) const {
    if (rows_ != cols_ || other.rows_ != other.cols_ || rows_ != other.rows_) {
        throw std::invalid_argument("Strassen's algorithm requires square matrices of equal size");
    }

    int n = rows_;

    if (n <= 64) {
        return multiply(other);
    }

    int new_size = n / 2;
    Matrix A11(new_size, new_size), A12(new_size, new_size), A21(new_size, new_size), A22(new_size, new_size);
    Matrix B11(new_size, new_size), B12(new_size, new_size), B21(new_size, new_size), B22(new_size, new_size);

    for (int i = 0; i < new_size; ++i) {
        for (int j = 0; j < new_size; ++j) {
            A11(i, j) = (*this)(i, j);
            A12(i, j) = (*this)(i, j + new_size);
            A21(i, j) = (*this)(i + new_size, j);
            A22(i, j) = (*this)(i + new_size, j + new_size);
            B11(i, j) = other(i, j);
            B12(i, j) = other(i, j + new_size);
            B21(i, j) = other(i + new_size, j);
            B22(i, j) = other(i + new_size, j + new_size);
        }
    }


    Matrix M1 = (A11 + A22).strassen_multiply(B11 + B22);
    Matrix M2 = (A21 + A22).strassen_multiply(B11);
    Matrix M3 = A11.strassen_multiply(B12 - B22);
    Matrix M4 = A22.strassen_multiply(B21 - B11);
    Matrix M5 = (A11 + A12).strassen_multiply(B22);
    Matrix M6 = (A21 - A11).strassen_multiply(B11 + B12);
    Matrix M7 = (A12 - A22).strassen_multiply(B21 + B22);


    Matrix C11 = M1 + M4 - M5 + M7;
    Matrix C12 = M3 + M5;
    Matrix C21 = M2 + M4;
    Matrix C22 = M1 - M2 + M3 + M6;


    Matrix result(n, n);
    for (int i = 0; i < new_size; ++i) {
        for (int j = 0; j < new_size; ++j) {
            result(i, j) = C11(i, j);
            result(i, j + new_size) = C12(i, j);
            result(i + new_size, j) = C21(i, j);
            result(i + new_size, j + new_size) = C22(i, j);
        }
    }

    return result;
}

Matrix standard_multiply(const Matrix &A, const Matrix &B) {
    return A.multiply(B);
}

Matrix strassen_multiply(const Matrix &A, const Matrix &B) {
    return A.strassen_multiply(B);
}

void multiply_region(const Matrix &A, const Matrix &B, Matrix &result, int start_row, int end_row) {
    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < B.getCols(); ++j) {
            result(i, j) = 0;
            for (int k = 0; k < A.getCols(); ++k) {
                result(i, j) += A(i, k) * B(k, j);
            }
        }
    }
}

Matrix multithreaded_standard_multiply(const Matrix &A, const Matrix &B, int num_threads) {
    if (A.getCols() != B.getRows()) {
        throw std::invalid_argument("Incompatible matrix dimensions");
    }

    Matrix result(A.getRows(), B.getCols());
    std::vector<std::thread> threads;

    int rows_per_thread = A.getRows() / num_threads;
    int start_row = 0;

    for (int i = 0; i < num_threads - 1; ++i) {
        int end_row = start_row + rows_per_thread;
        threads.emplace_back(multiply_region, std::ref(A), std::ref(B), std::ref(result), start_row, end_row);
        start_row = end_row;
    }


    threads.emplace_back(multiply_region, std::ref(A), std::ref(B), std::ref(result), start_row, A.getRows());


    for (auto &thread: threads) {
        thread.join();
    }

    return result;
}

Matrix strassen_multiply_threaded(const Matrix &A, const Matrix &B, int num_threads) {
    if (A.getRows() != A.getCols() || B.getRows() != B.getCols() || A.getRows() != B.getRows()) {
        throw std::invalid_argument("Strassen's algorithm requires square matrices of equal size");
    }

    int n = A.getRows();

    if (n <= 64 || num_threads <= 1) {
        return A.multiply(B);
    }


    int new_size = n / 2;
    Matrix A11(new_size, new_size), A12(new_size, new_size), A21(new_size, new_size), A22(new_size, new_size);
    Matrix B11(new_size, new_size), B12(new_size, new_size), B21(new_size, new_size), B22(new_size, new_size);

    for (int i = 0; i < new_size; ++i) {
        for (int j = 0; j < new_size; ++j) {
            A11(i, j) = A(i, j);
            A12(i, j) = A(i, j + new_size);
            A21(i, j) = A(i + new_size, j);
            A22(i, j) = A(i + new_size, j + new_size);
            B11(i, j) = B(i, j);
            B12(i, j) = B(i, j + new_size);
            B21(i, j) = B(i + new_size, j);
            B22(i, j) = B(i + new_size, j + new_size);
        }
    }

    int new_threads = num_threads / 7;
    if (new_threads < 1) new_threads = 1;


    std::vector<std::thread> threads;
    std::vector<Matrix> M(7, Matrix(new_size, new_size));

    auto compute_M = [&](int index, const Matrix &X, const Matrix &Y) {
        M[index] = strassen_multiply_threaded(X, Y, new_threads);
    };

    threads.emplace_back(compute_M, 0, A11 + A22, B11 + B22);
    threads.emplace_back(compute_M, 1, A21 + A22, B11);
    threads.emplace_back(compute_M, 2, A11, B12 - B22);
    threads.emplace_back(compute_M, 3, A22, B21 - B11);
    threads.emplace_back(compute_M, 4, A11 + A12, B22);
    threads.emplace_back(compute_M, 5, A21 - A11, B11 + B12);
    threads.emplace_back(compute_M, 6, A12 - A22, B21 + B22);


    for (auto &thread: threads) {
        thread.join();
    }


    Matrix C11 = M[0] + M[3] - M[4] + M[6];
    Matrix C12 = M[2] + M[4];
    Matrix C21 = M[1] + M[3];
    Matrix C22 = M[0] - M[1] + M[2] + M[5];


    Matrix result(n, n);
    for (int i = 0; i < new_size; ++i) {
        for (int j = 0; j < new_size; ++j) {
            result(i, j) = C11(i, j);
            result(i, j + new_size) = C12(i, j);
            result(i + new_size, j) = C21(i, j);
            result(i + new_size, j + new_size) = C22(i, j);
        }
    }

    return result;
}

Matrix multithreaded_strassen_multiply(const Matrix &A, const Matrix &B, int num_threads) {
    return strassen_multiply_threaded(A, B, num_threads);
}
