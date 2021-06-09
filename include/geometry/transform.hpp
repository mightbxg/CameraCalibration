#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <limits>

namespace bxg {

template <typename Scalar_ = double>
class RigidTransform {
public:
    using Scalar = Scalar_;
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
    using Vec6 = Eigen::Matrix<Scalar, 6, 1>;
    using Mat33 = Eigen::Matrix<Scalar, 3, 3>;
    using Mat36 = Eigen::Matrix<Scalar, 3, 6>;

    RigidTransform(const Mat33& rmat, const Vec3& tvec = Vec3::Zero())
        : rmat_(rmat)
        , tvec_(tvec)
    {
    }
    RigidTransform(const Vec3& rvec, const Vec3& tvec = Vec3::Zero())
        : rmat_(Eigen::AngleAxis<Scalar>(rvec.norm(), rvec.normalized()).toRotationMatrix())
        , tvec_(tvec)
    {
    }
    RigidTransform(const Vec6& param = Vec6::Zero())
        : RigidTransform(Vec3(param.template head<3>()), param.template tail<3>())
    {
    }

    static inline Mat33 skewSym(const Vec3& v)
    {
        Mat33 ret;
        ret << Scalar(0), -v[2], v[1],
            v[2], Scalar(0), -v[0],
            -v[1], v[0], Scalar(0);
        return ret;
    }

    // Jacobian: [d_rvec d_tvec]
    inline Vec3 transform(const Vec3& pt, Mat36* J_param = nullptr) const
    {
        Vec3 ret = rmat_ * pt + tvec_;
        if (J_param) {
            J_param->block(0, 0, 3, 3) = -skewSym(ret);
            J_param->block(0, 3, 3, 3) = Mat33::Identity();
        }
        return ret;
    }

public:
    Mat33 rmat_;
    Vec3 tvec_;
};

} //namespace bxg
