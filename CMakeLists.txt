cmake_minimum_required(VERSION 3.10)
project(ml_library)

set(CMAKE_CXX_STANDARD 11)

add_library(ml_library
    src/linear_algebra.cpp
    src/linear_regression.cpp
    src/k_means.cpp
    src/decision_tree.cpp
    src/classification.cpp
)

target_include_directories(ml_library PUBLIC include)

add_executable(test_ml_library
    tests/test_linear_regression.cpp
    tests/test_k_means.cpp
    tests/test_decision_tree.cpp
    tests/test_classification.cpp
)

target_link_libraries(test_ml_library ml_library)
