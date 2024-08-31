cmake_minimum_required(VERSION 3.28)
project(matrix_compare)

set(CMAKE_CXX_STANDARD 26)

add_executable(matrix_compare main.cpp
        matrix.h
        matrix.cpp
        )
