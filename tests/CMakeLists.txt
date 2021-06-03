if(NOT TARGET gtest_main)
    add_subdirectory(googletest EXCLUDE_FROM_ALL)
endif()

link_libraries(CameraCalibrator gtest gtest_main)

add_executable(test_camera src/test_camera.cpp)

include(GoogleTest)
gtest_discover_tests(test_camera)
