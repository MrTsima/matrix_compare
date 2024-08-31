#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <numeric>
#include "matrix.h"

const int NUM_TESTS = 10;

struct TestResult {
    std::vector<double> times;
    double avg, min, max;
};

TestResult runTest(const std::function<Matrix()> &testFunction) {
    TestResult result;
    for (int i = 0; i < NUM_TESTS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        testFunction();
        auto end = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double, std::milli>(end - start).count();
        result.times.push_back(time);
    }
    result.avg = std::accumulate(result.times.begin(), result.times.end(), 0.0) / NUM_TESTS;
    result.min = *std::min_element(result.times.begin(), result.times.end());
    result.max = *std::max_element(result.times.begin(), result.times.end());
    return result;
}

void printResults(const std::string &methodName, const TestResult &result) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << methodName << " multiplication times (ms):\n";
    std::cout << "  Average: " << result.avg << "\n";
    std::cout << "  Min: " << result.min << "\n";
    std::cout << "  Max: " << result.max << "\n\n";
}

int main() {
    /* rows and cols have to be the same to test strassen multiply,
     * but if you want to see results faster, make them smaller values.
     *
     * cons: advantage of strassen is not visible well with lower values.
     * */
    int rows = 1024;
    int cols = 1024;

    Matrix A(rows, cols);
    Matrix B(cols, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            A(i, j) = rand() % 100;
            B(i, j) = rand() % 100;
        }
    }

    int num_threads = std::thread::hardware_concurrency();
    std::cout << "Using " << num_threads << " threads.\n\n";

    auto standardResult = runTest([&]() { return standard_multiply(A, B); });
    auto strassenResult = runTest([&]() { return strassen_multiply(A, B); });
    auto multithreadedStandardResult = runTest([&]() { return multithreaded_standard_multiply(A, B, num_threads); });
    auto multithreadedStrassenResult = runTest([&]() { return multithreaded_strassen_multiply(A, B, num_threads); });


    printResults("Single-threaded standard", standardResult);
    printResults("Single-threaded Strassen", strassenResult);
    printResults("Multi-threaded standard", multithreadedStandardResult);
    printResults("Multi-threaded Strassen", multithreadedStrassenResult);


    Matrix result_standard = standard_multiply(A, B);
    Matrix result_strassen = strassen_multiply(A, B);
    Matrix result_multithreaded_standard = multithreaded_standard_multiply(A, B, num_threads);
    Matrix result_multithreaded_strassen = multithreaded_strassen_multiply(A, B, num_threads);

    if (result_standard == result_strassen &&
        result_standard == result_multithreaded_standard &&
        result_standard == result_multithreaded_strassen) {
        std::cout << "All results match.\n";
    } else {
        std::cout << "Results do not match.\n";
    }

    return 0;
}