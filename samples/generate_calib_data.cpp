#include "camera/brown_camera.hpp"
#include "geometry/transform.hpp"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
    if (argc < 2) {
        cout << "no input!\n";
        return 0;
    }

    // board config
    const int bd_rows = 6, bd_cols = 9;
    const float bd_step = 30.f;

    // load data
    Mat cam_mtx, dis_cef;
    Size image_size;
    vector<Vec3d> rvecs, tvecs;
    {
        FileStorage fs;
        if (!fs.open(argv[1], FileStorage::READ)) {
            cout << "cannot open " << argv[1] << " to read\n";
            return -1;
        }
#define LOAD_FS(val) fs[#val] >> val;
        LOAD_FS(cam_mtx);
        LOAD_FS(dis_cef);
        LOAD_FS(image_size);
        LOAD_FS(rvecs);
        LOAD_FS(tvecs);
#undef LOAD_FS
        CV_Assert(cam_mtx.type() == CV_64FC1);
        CV_Assert(dis_cef.type() == CV_64FC1);
        CV_Assert(image_size.area() > 0);
        CV_Assert(rvecs.size() == tvecs.size());
    }
    const size_t num_frame = rvecs.size();
    cout << "frame num: " << num_frame << endl;

    // construct camera
    using Camera = bxg::BrownCamera<double>;
    Camera::VecN params;
    {
        const double* pi = cam_mtx.ptr<double>();
        const double* pd = dis_cef.ptr<double>();
        params << pi[0], pi[4], pi[2], pi[5],
            pd[0], pd[1], pd[4], pd[2], pd[3];
    }
    Camera cam(params);

    // control points
    vector<Camera::Vec3> pts3d;
    for (int y = 0; y < bd_rows; ++y)
        for (int x = 0; x < bd_cols; ++x)
            pts3d.emplace_back(x * bd_step, y * bd_step, 0.0);

    // rvec, tvec => RigidTransform
    using Transform = bxg::RigidTransform<double>;
    auto getRt = [](const Vec3d& r, const Vec3d& t) -> Transform {
        Transform::Vec6 params;
        params << r[0], r[1], r[2], t[0], t[1], t[2];
        return Transform(params);
    };

    // generate pts
    vector<vector<Point3f>> vpts3d_cv;
    vector<vector<Point2f>> vpts2d_cv;
    for (size_t idx_frame = 0; idx_frame < num_frame; ++idx_frame) {
        auto rt = getRt(rvecs[idx_frame], tvecs[idx_frame]);
        vector<Point3f> pts3d_cv;
        vector<Point2f> pts2d_cv;
        for (const auto& pt3d : pts3d) {
            Camera::Vec2 pt2d;
            if (cam.project(rt.transform(pt3d), pt2d)) {
                pts3d_cv.emplace_back(pt3d.x(), pt3d.y(), pt3d.z());
                pts2d_cv.emplace_back(pt2d.x(), pt2d.y());
            } else {
                printf("\33[31mproject failed: %f %f %f\33[0m\n", pt3d.x(), pt3d.y(), pt3d.z());
            }
        }
        vpts3d_cv.push_back(pts3d_cv);
        vpts2d_cv.push_back(pts2d_cv);
    }

    // draw frames
    Mat image_bg = Mat(image_size, CV_8UC3, Scalar::all(255));
    for (size_t i = 0; i < vpts2d_cv.size(); ++i) {
        Mat image = image_bg.clone();
        for (const auto& pt : vpts2d_cv[i])
            circle(image, pt, 2, { 0, 0, 255 }, cv::FILLED);
        imwrite(format("%02zu.png", i), image);
    }

    // save pts
    {
        FileStorage fs;
        string fn_out = "pts.yaml";
        if (!fs.open(fn_out, FileStorage::WRITE)) {
            cout << "cannot open " << fn_out << " to write" << endl;
            return -1;
        }
        fs << "vpts3d" << vpts3d_cv;
        fs << "vpts2d" << vpts2d_cv;
    }

    cout << "all done" << endl;
    return 0;
}
