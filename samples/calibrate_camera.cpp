#include "estimator/camera_calibrator.h"
#include <iostream>
#include <opencv2/core.hpp>

using namespace std;

namespace cv {

template <typename _Scalar, int _Dim>
class DataType<Eigen::Matrix<_Scalar, _Dim, 1>> {
public:
    typedef Eigen::Matrix<_Scalar, _Dim, 1> value_type;
    typedef Eigen::Matrix<typename DataType<_Scalar>::work_type, _Dim, 1> work_type;
    typedef _Scalar channel_type;

    enum { generic_type = 0,
        channels = _Dim,
        fmt = traits::SafeFmt<channel_type>::fmt + ((channels - 1) << 8)
    };

    typedef Vec<channel_type, channels> vec_type;
};

} //namespace cv

int main(int argc, char* argv[])
{
    // load data from file
    if (argc < 2) {
        cout << "no input\n";
        return 0;
    }
    vector<vector<Eigen::Vector3d>> vpts3d;
    vector<vector<Eigen::Vector2d>> vpts2d;
    {
        cv::FileStorage fs;
        string fn(argv[1]);
        if (!fs.open(fn, cv::FileStorage::READ)) {
            cout << "cannot open " << fn << '\n';
            return -1;
        }
        fs["vpts3d"] >> vpts3d;
        fs["vpts2d"] >> vpts2d;
        CV_Assert(vpts3d.size() == vpts2d.size());
    }

    vector<double> params = { 631.6558837890625, 632.6558837890625, 637.0838623046875, 390.05694580078125,
        -0.03983112892232504, 0.03768857717889241, 0.0, 0.0007003028120895412, -0.0032084429231644357 };
    auto success = bxg::CameraCalibrator::optimize(vpts3d, vpts2d, params);
    cout << "result:\n";
    for (double v : params)
        cout << v << " ";
    cout << endl;

    return 0;
}
