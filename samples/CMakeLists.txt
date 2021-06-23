add_executable(generate_calib_data generate_calib_data.cpp)
target_link_libraries(generate_calib_data CameraCalibrator)

add_executable(calibrate_camera calibrate_camera.cpp)
target_link_libraries(calibrate_camera CameraCalibrator)

set(FRONTEND_LIB "" CACHE STRING "Frontend library")
if(FRONTEND_LIB)
    find_package(CalibLink REQUIRED)
    add_executable(calibrate_with_image calibrate_with_image.cpp)
    target_link_libraries(calibrate_with_image
        CameraCalibrator OBStarCalib::CalibLink ${FRONTEND_LIB})
endif()
