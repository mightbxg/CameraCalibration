if(NOT TARGET gtest_main)
    add_subdirectory(googletest EXCLUDE_FROM_ALL)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
link_libraries(CameraCalibrator gtest gtest_main)

add_executable(test_camera src/test_camera.cpp)
target_link_libraries(test_camera PRIVATE ${OpenCV_LIBS})
add_executable(test_geometry src/test_geometry.cpp)
target_link_libraries(test_geometry PRIVATE ${OpenCV_LIBS})

include(GoogleTest)
gtest_discover_tests(test_camera)
gtest_discover_tests(test_geometry)
