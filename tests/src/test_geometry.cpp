#include "geometry/transform.hpp"
#include "test_utils.hpp"
#include "gtest/gtest.h"
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <random>

namespace {

template <typename Scalar>
std::vector<Eigen::Matrix<Scalar, 7, 1>> getTestPoses(size_t num = 1000)
{
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
    using Vec7 = Eigen::Matrix<Scalar, 7, 1>;
    std::vector<Vec7> ret;

    std::default_random_engine re;
    std::uniform_real_distribution<> dis_axis(-10.0, 10.0);
    std::uniform_real_distribution<> dis_theta(-M_PI, M_PI);
    std::uniform_int_distribution<> dis_coord(-5, 5);
    ret.reserve(num);
    for (size_t i = 0; i < num; ++i) {
        Vec7 p;
        Eigen::AngleAxis<Scalar> rvec(dis_theta(re), Vec3(dis_axis(re), dis_axis(re), dis_axis(re)).normalized());
        p.template tail<4>() = Eigen::Quaternion<Scalar>(rvec).coeffs();
        p.template head<3>() = Vec3(dis_coord(re), dis_coord(re), dis_coord(re));
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
    using TransT = bxg::RigidTransform<Scalar>;
    using Vec3 = typename TransT::Vec3;
    using VecN = typename TransT::VecN;
    using Mat3N = typename TransT::Mat3N;
    auto poses = getTestPoses<Scalar>(500);
    auto pts = getTestPts<Scalar>();

    const Scalar eps = TestConstants<Scalar>::epsilon;

    for (const auto& pose : poses) {
        bxg::RigidTransform<Scalar> rt(pose);
        cv::Mat tvec({ pose[0], pose[1], pose[2] });
        Eigen::Quaternion<Scalar> quaternion(pose.template tail<4>());
        cv::Mat rmat;
        cv::eigen2cv(quaternion.toRotationMatrix(), rmat);

        for (const auto& pt : pts) {
            cv::Mat pt_cv({ pt.x(), pt.y(), pt.z() });
            cv::Mat pt_cv_dst = rmat * pt_cv + tvec;
            auto pt_ref = pt_cv_dst.ptr<Vec3>()[0];

            Mat3N J_param;
            auto pt_dst = rt.transform(pt, &J_param(0, 0));
            ASSERT_TRUE(isApprox(pt_ref, pt_dst, eps))
                << "expect: " << pt_ref.transpose()
                << "\nresult: " << pt_dst.transpose();

            test_jacobian(
                "J_param", J_param, [&](const VecN& p) {
                    auto _rt = rt + p;
                    return _rt.transform(pt);
                },
                VecN::Zero());
        }
    }
}

TEST(Geometry, RigidTransformDouble)
{
    testRigidTransform<double>();
}

TEST(Geometry, RigidTransformFloat)
{
    testRigidTransform<float>();
}

} //anonymous namespace
