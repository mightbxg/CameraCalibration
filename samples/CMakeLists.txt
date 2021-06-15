add_executable(generate_calib_data generate_calib_data.cpp)
target_link_libraries(generate_calib_data CameraCalibrator ${OpenCV_LIBS})

add_executable(calibrate calibrate.cpp)
target_link_libraries(calibrate CameraCalibrator ${OpenCV_LIBS})
