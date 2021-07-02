#include "estimator/camera_calibrator.h"
#include <CalibLink/CalibTarget.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace obstarcalib;

namespace {
template <typename T>
ostream& operator<<(ostream& os, const vector<T>& vec)
{
    for (const auto& v : vec)
        os << v << " ";
    return os;
}
}

int main(int argc, char* argv[])
{
    printf("CalibLink version[%d.%d.%d]\n",
        CalibLink_VERSION_MAJOR,
        CalibLink_VERSION_MINOR,
        CalibLink_VERSION_PATCH);
    printf("Frontend[%s] version[%d.%d.%d]\n\n",
        Frontend_NAME(),
        Frontend_VERSION_MAJOR(),
        Frontend_VERSION_MINOR(),
        Frontend_VERSION_PATCH());

    if (argc < 3) {
        cout << "Usage: [config] [image]\n";
        return 0;
    }
    // load config and source image
    string fn_config = argv[1];
    string fn_image = argv[2];
    auto target = caliblink::CalibTarget::getInstance();
    if (!target->loadConfig(fn_config)) {
        cout << "cannot load config from: " << fn_config << '\n';
        return -1;
    }
    Mat image_src = imread(fn_image, IMREAD_GRAYSCALE);
    if (image_src.empty()) {
        cout << "cannot load image from: " << fn_image << '\n';
        return -1;
    }

    vector<caliblink::CtrlPointd> cpts;
    if (!target->detect(image_src, cpts)) {
        cout << "detect target from image failed\n";
        return -2;
    }
    Mat image_board;
    target->draw(image_src, image_board, 5.0);
    imwrite("boards.png", image_board);

    // construct pts
    vector<vector<Eigen::Vector3d>> vpts3d;
    vector<vector<Eigen::Vector2d>> vpts2d;
    vpts3d.resize(4);
    vpts2d.resize(4);
    for (const auto& cpt : cpts) {
        auto info = target->ptInfo(cpt.index());
        vpts3d[info.planeId].emplace_back(info.pt3d);
        vpts2d[info.planeId].emplace_back(cpt.x(), cpt.y());
    }

    // calibrate
    vector<double> params = { 100, 100, 80, 60, 0, 0, 0, 0, 0 };
    vector<double> covariance;
    vector<bxg::CameraCalibrator::TransformParams> transforms;
    bxg::CameraCalibrator solver;
    solver.options.minimizer_progress_to_stdout = true;
    solver.options.report_type = bxg::ReportType::FULL;
    auto errs = solver.optimize(vpts3d, vpts2d, params, &covariance, &transforms);
    cout << "result: " << params << '\n';
    cout << "covari: " << covariance << '\n';
    printf("errors: min[%f] max[%f] avg[%f]\n", errs[0], errs[1], errs[2]);

    cout << "calib with direct method-------------------------\n";
    //solver.options.minimizer_progress_to_stdout = false;
    //solver.options.report_type = bxg::ReportType::NONE;
    solver.optimize(image_src, params, &covariance);
    cout << "result: " << params << '\n';
    cout << "covari: " << covariance << '\n';

    return 0;
}
