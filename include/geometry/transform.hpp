#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <limits>

namespace bxg {

#define USE_PERTURBATION_MODEL 0

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

    // Jacobian: [d_rvec d_tvec]
    inline Vec3 transform(const Vec3& pt, Scalar* J_param = nullptr) const
    {
        Vec3 ret = rmat_ * pt + tvec_;
        if (J_param) {
            Eigen::Map<Mat3N> jac(J_param);
#if USE_PERTURBATION_MODEL
            jac.block(0, 0, 3, 3) = -skewSym(ret);
#else
            Eigen::AngleAxis<Scalar> aa;
            aa.fromRotationMatrix(rmat_);
            Scalar v1, v2; // v1=sinx/x v2=(1-cosx)/x
            if (abs(aa.angle()) < Eigen::NumTraits<Scalar>::dummy_precision()) {
                v1 = Scalar(1);
                v2 = Scalar(0);
            } else {
                v1 = sin(aa.angle()) / aa.angle();
                v2 = (Scalar(1) - cos(aa.angle())) / aa.angle();
            }
            Mat33 J_l = v1 * Mat33::Identity()
                + (Scalar(1) - v1) * aa.axis() * aa.axis().transpose()
                + v2 * skewSym(aa.axis());
            jac.block(0, 0, 3, 3).noalias() = -skewSym(rmat_ * pt) * J_l;
#endif
            jac.block(0, 3, 3, 3) = Mat33::Identity();
        }
        return ret;
    }

    const RigidTransform operator+(const VecN& p) const
    {
#if USE_PERTURBATION_MODEL
        auto R = toRotationMatrix(p.template head<3>());
        auto t = p.template tail<3>();
        return RigidTransform(R * rmat_, R * tvec_ + t);
#else
        return RigidTransform(params() + p);
#endif
    }

    VecN params() const
    {
        VecN p;
        p.template head<3>() = toRotationVector(rmat_);
        p.template tail<3>() = tvec_;
        return p;
    }

private:
    Mat33 rmat_;
    Vec3 tvec_;
};

} //namespace bxg
