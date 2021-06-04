cmake_minimum_required(VERSION 3.10)

project(CameraCalibration)

# dependencies
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)

# main library
add_library(CameraCalibrator INTERFACE)
target_include_directories(CameraCalibrator INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(CameraCalibrator INTERFACE cxx_std_17)
target_compile_options(CameraCalibrator INTERFACE
    "-Wall;-Wextra;-Wunused;-Wformat=2;-Wshadow")
target_link_libraries(CameraCalibrator INTERFACE Eigen3::Eigen Ceres::ceres)

# tests
enable_testing()
add_subdirectory(tests)
