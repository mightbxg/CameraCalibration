#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <limits>

namespace bxg {

/// @brief Rigid transform with rotation represented by quaternion
template <typename Scalar_ = double>
class QuaternionTransform {
public:
    static constexpr int N = 7; //!< tx, ty, tz, qx, qy, qz, qw
    using Scalar = Scalar_;
    using Quaternion = Eigen::Quaternion<Scalar>;
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
    using Vec4 = Eigen::Matrix<Scalar, 4, 1>;
    using VecN = Eigen::Matrix<Scalar, N, 1>;
    using Mat33 = Eigen::Matrix<Scalar, 3, 3, Eigen::RowMajor>;
    using Mat3N = Eigen::Matrix<Scalar, 3, N, Eigen::RowMajor>;

    QuaternionTransform(const Mat33& rmat, const Vec3& tvec)
        : rmat_(rmat)
        , tvec_(tvec)
    {
    }
    QuaternionTransform(const VecN& param = VecN::Zero())
        : QuaternionTransform(Quaternion(param.template tail<4>()).toRotationMatrix(), param.template head<3>())
    {
    }

    static inline Mat33 toRotationMatrix(const Vec3& rvec)
    {
        return Eigen::AngleAxis<Scalar>(rvec.norm(), rvec.normalized()).toRotationMatrix();
    }

    static inline Vec3 toRotationVector(const Mat33& rmat)
    {
        Eigen::AngleAxis<Scalar> aa;
        aa.fromRotationMatrix(rmat);
        return aa.axis() * aa.angle();
    }

    static inline Mat33 skewSym(const Vec3& v)
    {
        Mat33 ret;
        ret << Scalar(0), -v[2], v[1],
            v[2], Scalar(0), -v[0],
            -v[1], v[0], Scalar(0);
        return ret;
    }

    // Jacobian: [d_tvec d_rvec]
    inline Vec3 transform(const Vec3& pt, Scalar* J_param = nullptr) const
    {
        Vec3 ret = rmat_ * pt + tvec_;
        if (J_param) {
            Eigen::Map<Mat3N> jac(J_param);
            jac.block(0, 0, 3, 3).setIdentity();
            jac.block(0, 3, 3, 3).noalias() = -rmat_ * skewSym(pt);
            jac.template rightCols<1>().setZero();
        }
        return ret;
    }

    const QuaternionTransform operator+(const VecN& p) const
    {
        auto R = toRotationMatrix(p.template segment<3>(3));
        auto t = p.template head<3>();
        return QuaternionTransform(rmat_ * R, t + tvec_);
    }

    VecN params() const
    {
        VecN p;
        p.template tail<4>() = Quaternion(rmat_).coeffs();
        p.template head<3>() = tvec_;
        return p;
    }

private:
    Mat33 rmat_;
    Vec3 tvec_;
};

} //namespace bxg
