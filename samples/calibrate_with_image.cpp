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

bool writeIntrinsicParams(const string& filename, const vector<double>& params, const cv::Size& image_size)
{
    FileStorage fs;
    if (!fs.open(filename, FileStorage::WRITE))
        return false;
    CV_Assert(params.size() == 9);

    fs.writeComment("projection");
    fs << "fx" << params[0];
    fs << "fy" << params[1];
    fs << "cx" << params[2];
    fs << "cy" << params[3];

    fs.writeComment("distortion");
    fs << "k1" << params[4];
    fs << "k2" << params[5];
    fs << "k3" << params[6];
    fs << "p1" << params[7];
    fs << "p2" << params[8];

    fs.writeComment("image size");
    fs << "width" << image_size.width;
    fs << "height" << image_size.height;

    return true;
}

bool writeExtrinsicParams(const string& filename,
    const vector<bxg::CameraCalibrator::TransformParams>& transforms)
{
    FileStorage fs;
    if (!fs.open(filename, FileStorage::WRITE))
        return false;
    vector<vector<double>> poses;
    for (const auto& tran : transforms) {
        vector<double> pose;
        for (int i = 0; i < tran.size(); ++i)
            pose.emplace_back(tran[i]);
        poses.emplace_back(pose);
    }
    fs << "poses" << poses;
    return true;
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
    const bool is_low_res = image_src.cols == 160;

    vector<caliblink::CtrlPointd> cpts;
    if (!target->detect(image_src, cpts)) {
        cout << "detect target from image failed\n";
        return -2;
    }
    Mat image_kps;
    target->draw(image_src, image_kps, is_low_res ? 5.0 : 1.0);
    imwrite("kps.png", image_kps);

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
    vector<double> params = { 500, 500, 320, 240, 0, 0, 0, 0, 0 };
    if (is_low_res) {
        for (auto& p : params)
            p /= 4.0;
    }
    vector<double> covariance;
    vector<bxg::CameraCalibrator::TransformParams> transforms;
    bxg::CameraCalibrator solver;
    solver.options.minimizer_progress_to_stdout = true;
    solver.options.report_type = bxg::ReportType::FULL;
    auto errs = solver.optimize(vpts3d, vpts2d, params, &covariance, &transforms);
    cout << "result: " << params << '\n';
    cout << "covari: " << covariance << '\n';
    printf("errors: min[%f] max[%f] avg[%f]\n", errs[0], errs[1], errs[2]);
    Mat image_sim(image_src.size(), CV_8UC1);
    solver.drawSimBoard(image_sim, params, transforms);
    imwrite("restored_kps.png", image_sim);
    Mat image_balanced;
    solver.balanceImage(image_src, image_balanced, params, transforms);
    imwrite("balanced.png", image_balanced);

    cout << "\33[32mcalib with direct method-------------------------\33[0m\n";
    solver.optimize(image_src, params, &covariance, &transforms);
    cout << "result: " << params << '\n';
    cout << "covari: " << covariance << '\n';
    solver.drawSimBoard(image_sim, params, transforms);
    imwrite("restored_direct.png", image_sim);

    // output result
    writeIntrinsicParams("camera_intrinsic.yml", params, image_src.size());
    writeExtrinsicParams("board_poses.yml", transforms);

    return 0;
}
