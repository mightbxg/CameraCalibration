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

template <typename T>
ostream& operator<<(ostream& os, const vector<T>& vec)
{
    for (const auto& v : vec)
        os << v << " ";
    return os;
}

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

    vector<double> params_ref = { 532.827, 532.946, 342.487, 233.856, -0.280881, 0.0251717, 0.163449, 0.00121657, -0.000135549 };
    vector<double> params = { 500.0, 500.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    bxg::CameraCalibrator::optimize(vpts3d, vpts2d, params);

    cout << "refere: " << params_ref << '\n';
    cout << "result: " << params << '\n';

    return 0;
}
