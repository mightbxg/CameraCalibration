#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <limits>

namespace bxg {

template <typename Scalar_ = double>
class RigidTransform {
public:
    static constexpr int N = 6; //!< rx, ry, rz, tx, ty, tz
    using Scalar = Scalar_;
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
    using VecN = Eigen::Matrix<Scalar, N, 1>;
    using Mat33 = Eigen::Matrix<Scalar, 3, 3, Eigen::RowMajor>;
    using Mat3N = Eigen::Matrix<Scalar, 3, N, Eigen::RowMajor>;

    RigidTransform(const Mat33& rmat, const Vec3& tvec)
        : rmat_(rmat)
        , tvec_(tvec)
    {
    }
    RigidTransform(const VecN& param = VecN::Zero())
        : RigidTransform(toRotationMatrix(param.template head<3>()), param.template tail<3>())
    {
    }

    static inline Mat33 toRotationMatrix(const Vec3& rvec)
    {
        return Eigen::AngleAxis<Scalar>(rvec.norm(), rvec.normalized()).toRotationMatrix();
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
    inline Vec3 transform(const Vec3& pt, Scalar* J_param = nullptr) const
    {
        Vec3 ret = rmat_ * pt + tvec_;
        if (J_param) {
            Eigen::Map<Mat3N> jac(J_param);
            jac.block(0, 0, 3, 3) = -skewSym(ret);
            jac.block(0, 3, 3, 3) = Mat33::Identity();
        }
        return ret;
    }

    const RigidTransform operator+(const VecN& p) const
    {
        auto R = toRotationMatrix(p.template head<3>());
        auto t = p.template tail<3>();
        return RigidTransform(R * rmat_, R * tvec_ + t);
    }

public:
    Mat33 rmat_;
    Vec3 tvec_;
};

} //namespace bxg
