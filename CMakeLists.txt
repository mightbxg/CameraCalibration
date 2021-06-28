cmake_minimum_required(VERSION 3.10)

project(CameraCalibration)

# dependencies
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)

# main library
file(GLOB_RECURSE HEADERS_FOR_IDE
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")
add_library(CameraCalibrator
    src/camera_calibrator.cpp
    ${HEADERS_FOR_IDE})
target_include_directories(CameraCalibrator PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(CameraCalibrator PUBLIC cxx_std_17)
target_compile_options(CameraCalibrator PUBLIC
    "-Wall;-Wextra;-Wunused;-Wformat=2;-Wshadow")
target_link_libraries(CameraCalibrator
    PUBLIC Eigen3::Eigen ${OpenCV_LIBS} Sophus::Sophus
    PRIVATE Ceres::ceres)

# samples
add_subdirectory(samples)

# tests
enable_testing()
add_subdirectory(tests)
