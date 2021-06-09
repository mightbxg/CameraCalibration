#include "geometry/transform.hpp"
#include "test_utils.hpp"
#include "gtest/gtest.h"
#include <opencv2/opencv.hpp>
#include <random>

namespace {

template <typename Scalar>
std::vector<Eigen::Matrix<Scalar, 6, 1>> getTestPoses(size_t num = 1000)
{
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
    using Vec6 = Eigen::Matrix<Scalar, 6, 1>;
    std::vector<Vec6> ret;

    std::default_random_engine re;
    std::uniform_real_distribution<> dis_axis(-10.0, 10.0);
    std::uniform_real_distribution<> dis_theta(-M_PI, M_PI);
    std::uniform_int_distribution<> dis_coord(-5, 5);
    ret.reserve(num);
    for (size_t i = 0; i < num; ++i) {
        Vec6 p;
        p.template head<3>() = Vec3(dis_axis(re), dis_axis(re), dis_axis(re)).normalized() * dis_theta(re);
        p.template tail<3>() = Vec3(dis_coord(re), dis_coord(re), dis_coord(re));
        ret.push_back(p);
    }
    return ret;
}

template <typename Scalar>
std::vector<Eigen::Matrix<Scalar, 3, 1>> getTestPts()
{
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
    std::vector<Vec3> ret;
    for (int x = -5; x <= 5; ++x)
        for (int y = -5; y <= 5; ++y)
            for (int z = -5; z <= 5; ++z)
                ret.emplace_back(x, y, z);
    return ret;
}

template <typename Scalar>
void testRigidTransform()
{
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
    auto poses = getTestPoses<Scalar>();
    auto pts = getTestPts<Scalar>();

    const Scalar eps = TestConstants<Scalar>::epsilon;

    for (const auto& pose : poses) {
        bxg::RigidTransform<Scalar> rt(pose);
        cv::Mat rvec({ pose[0], pose[1], pose[2] });
        cv::Mat tvec({ pose[3], pose[4], pose[5] });
        cv::Mat rmat;
        cv::Rodrigues(rvec, rmat);

        for (const auto& pt : pts) {
            cv::Mat pt_cv({ pt.x(), pt.y(), pt.z() });
            cv::Mat pt_cv_dst = rmat * pt_cv + tvec;
            auto pt_ref = pt_cv_dst.ptr<Vec3>()[0];

            auto pt_dst = rt.transform(pt);
            ASSERT_TRUE(isApprox(pt_ref, pt_dst, eps))
                << "expect: " << pt_ref.transpose()
                << "\nresult: " << pt_dst.transpose();
        }
    }
}

TEST(Geometry, RigidTransformDouble)
{
    testRigidTransform<double>();
}

} //anonymous namespace
