#include "camera/brown_camera.hpp"
#include "test_utils.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "gtest/gtest.h"

namespace {

template <typename CamT>
void testProjectUnproject()
{
    auto cams = CamT::getTestProjections();
    using Vec2 = typename CamT::Vec2;
    using Vec3 = typename CamT::Vec3;

    auto toVec2 = [](const cv::Point2d& pt) -> Vec2 {
        return { pt.x, pt.y };
    };
    auto toVec3 = [](const cv::Point3d& pt) -> Vec3 {
        return { pt.x, pt.y, pt.z };
    };

    std::vector<cv::Point3d> pts3d;
    for (int x = -10; x <= 10; ++x)
        for (int y = -10; y <= 10; ++y)
            for (int z = 0; z < 5; ++z)
                pts3d.emplace_back(x, y, z);

    for (const auto& cam : cams) {
        const auto& p = cam.param();
        cv::Mat cam_mtx = (cv::Mat_<double>(3, 3) << p[0], 0.0, p[2],
            0.0, p[1], p[3], 0.0, 0.0, 1.0);
        cv::Mat dis_cef = (cv::Mat_<double>(5, 1) << p[4], p[5], p[7], p[8], p[6]);
        cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
        cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);
        std::vector<cv::Point2d> pts2d;
        cv::projectPoints(pts3d, rvec, tvec, cam_mtx, dis_cef, pts2d);

        for (size_t i = 0; i < pts3d.size(); ++i) {
            Vec2 pt2d;
            bool success = cam.project(toVec3(pts3d[i]), pt2d);
            if (success) {
                auto pt2d_ref = toVec2(pts2d[i]);
                EXPECT_TRUE(pt2d.isApprox(pt2d_ref))
                    << "expect: " << pt2d_ref.transpose()
                    << "\nresult: " << pt2d.transpose();
            }
        }
    }
}

template <typename CamT>
void testProjectJacobian()
{
    auto cams = CamT::getTestProjections();
    using Vec2 = typename CamT::Vec2;
    using Vec3 = typename CamT::Vec3;
    using Mat23 = typename CamT::Mat23;
    using Mat2N = typename CamT::Mat2N;

    std::vector<Vec3> pts3d;
    for (int x = -10; x <= 10; ++x)
        for (int y = -10; y <= 10; ++y)
            for (int z = 0; z < 5; ++z)
                pts3d.emplace_back(x, y, z);

    for (const auto& cam : cams) {
        for (const auto& pt3d : pts3d) {
            Vec2 pt2d;
            Mat23 J_p;
            bool success = cam.project(pt3d, pt2d, &J_p);
            if (success) {
                test_jacobian(
                    "d_r_d_p", J_p, [&](const Vec3& x) {
                        Vec2 res;
                        cam.project(pt3d + x, res);
                        return res;
                    },
                    Vec3::Zero());
                break; //FIXME
            }
        }
    }
}

TEST(Camera, BrownProjectUnprojectFloat)
{
    testProjectUnproject<bxg::BrownCamera<float>>();
}

TEST(Camera, BrownProjectUnprojectDouble)
{
    testProjectUnproject<bxg::BrownCamera<double>>();
}

TEST(Camera, BrownProjectJacobian)
{
    testProjectJacobian<bxg::BrownCamera<double>>();
}

}
