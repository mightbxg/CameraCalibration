#pragma once

#include "camera/brown_camera.hpp"
#include "geometry/transform.hpp"
#include <ceres/ceres.h>
#include <geometry/chessboard.hpp>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

namespace bxg {
using CameraType = BrownCamera<double>;
using TransformType = QuaternionTransform<double>;

template <typename Derived, std::enable_if_t<Derived::ColsAtCompileTime == 1 && Derived::RowsAtCompileTime == 3, bool> = true>
inline static auto deltaQ(const Eigen::MatrixBase<Derived>& rvec)
{
    using Scalar = typename Derived::Scalar;
    Eigen::Matrix<Scalar, 4, 1> v;
    v.template head<3>() = rvec / Scalar(2);
    v[3] = Scalar(1);
    return Eigen::Quaternion<Scalar>(v);
}

class PoseLocalParameterization : public ceres::LocalParameterization {
public:
    virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const override
    {
        using VecN = TransformType::VecN;
        using Vec6 = Eigen::Matrix<double, 6, 1>;
        using Quaternion = TransformType::Quaternion;

        Eigen::Map<const VecN> params(x);
        Eigen::Map<const Vec6> del(delta);
        Eigen::Map<VecN> params_p_delta(x_plus_delta);

        Quaternion q(params.head<4>());
        Quaternion dq = deltaQ(del.head<3>());
        params_p_delta.head<4>() = (q * dq).normalized().coeffs();

        params_p_delta.tail<3>() = del.tail<3>() + params.tail<3>();
        return true;
    }
    virtual bool ComputeJacobian(const double* /*x*/, double* jacobian) const override
    {
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> jac(jacobian);
        jac.setZero();
        jac.block(0, 0, 3, 3).setIdentity();
        jac.block(4, 3, 3, 3).setIdentity();
        return true;
    }
    virtual int GlobalSize() const override { return 7; }
    virtual int LocalSize() const override { return 6; }
};

class Se3LocalParameterization : public ceres::LocalParameterization {
public:
    using Group = Sophus::SE3d;
    using Tangent = Group::Tangent;

    virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const override
    {
        Eigen::Map<const Group> T(x);
        Eigen::Map<const Tangent> dx(delta);
        Eigen::Map<Group> Tdx(x_plus_delta);
        Tdx = T * Group::exp(dx);
        return true;
    }
    virtual bool ComputeJacobian(const double* x, double* jacobian) const override
    {
        Eigen::Map<const Group> T(x);
        Eigen::Map<Eigen::Matrix<double, Group::num_parameters, Group::DoF, Eigen::RowMajor>> jac(jacobian);
        jac = T.Dx_this_mul_exp_x_at_0();
        return true;
    }
    virtual int GlobalSize() const override { return Group::num_parameters; }
    virtual int LocalSize() const override { return Group::DoF; }
};

struct CameraTransform {
public:
    template <typename T>
    using Vec2 = Eigen::Matrix<T, 2, 1>;
    template <typename T>
    using Vec3 = Eigen::Matrix<T, 3, 1>;

    /// project world point to the image
    template <typename T>
    static bool project(const T* const cam_params, const T* const trans_params,
        const Vec3<T>& pt_world, Vec2<T>& pt_image)
    {
        using CamT = BrownCamera<T>;
        auto camera = CamT(Eigen::Map<const typename CamT::VecN>(cam_params));
        Eigen::Map<const Sophus::SE3<T>> Tcw(trans_params);

        bool ret = camera.project(Tcw * pt_world, pt_image);
        return ret;
    }

    /// unproject image point to the board
    template <typename T>
    static bool unproject(const T* const cam_params, const T* const trans_params,
        const Vec2<T>& pt_image, Vec3<T>& pt_board)
    {
        using CamT = BrownCamera<T>;
        auto camera = CamT(Eigen::Map<const typename CamT::VecN>(cam_params));
        Eigen::Map<const Sophus::SE3<T>> Tcw(trans_params);

        Vec3<T> pt3d_cam;
        if (camera.unproject(pt_image, pt3d_cam)) {
            Sophus::SE3<T> Twc = Tcw.inverse();
            Vec3<T> r2 = Twc.rotationMatrix().row(2);
            T z_cam = -Twc.translation().z() / (r2.transpose() * pt3d_cam);
            pt_board = Twc * (z_cam * pt3d_cam);
            return true;
        }
        return false;
    }
};

struct ProjectCostFunctor {
public:
    using Vec2 = Eigen::Vector2d;
    using Vec3 = Eigen::Vector3d;

    ProjectCostFunctor(const Vec3& pt3d, const Vec2& pt2d)
        : pt3d_(pt3d)
        , pt2d_(pt2d)
    {
    }

    template <typename T>
    bool operator()(const T* const cam_params,
        const T* const trans_params, T* residual) const
    {
        Eigen::Matrix<T, 2, 1> pt2d_proj;
        bool ret = CameraTransform::project<T>(cam_params, trans_params, pt3d_.cast<T>(), pt2d_proj);
        residual[0] = pt2d_proj[0] - T(pt2d_[0]);
        residual[1] = pt2d_proj[1] - T(pt2d_[1]);
        return ret;
    }

private:
    const Vec3 pt3d_;
    const Vec2 pt2d_;
};

class ProjectCostFunction : public ceres::SizedCostFunction<2, CameraType::N, TransformType::N> {
public:
    using Vec2 = CameraType::Vec2;
    using Vec3 = CameraType::Vec3;
    using CameraParams = CameraType::VecN;
    using TransformParams = TransformType::VecN;

    ProjectCostFunction(const Vec3& pt3d, const Vec2& pt2d)
        : pt3d_(pt3d)
        , pt2d_(pt2d)
    {
    }

    virtual bool Evaluate(const double* const* parameters,
        double* residuals, double** jacobians) const override
    {
        auto camera = CameraType(Eigen::Map<const CameraParams>(parameters[0]));
        auto trans = TransformType(Eigen::Map<const TransformParams>(parameters[1]));

        double* ptr_J_cam = nullptr;
        TransformType::Mat3N d_pt3d_d_trans;
        CameraType::Mat23 d_proj_d_pt3d;
        double* ptr_d_pt3d_d_trans = nullptr;
        double* ptr_d_proj_d_pt3d = nullptr;
        if (jacobians) {
            ptr_J_cam = jacobians[0];
            ptr_d_pt3d_d_trans = &d_pt3d_d_trans(0, 0);
            ptr_d_proj_d_pt3d = &d_proj_d_pt3d(0, 0);
        }

        auto pt3d_cam = trans.transform(pt3d_, ptr_d_pt3d_d_trans);
        Vec2 pt2d_proj;
        bool ret = camera.project(pt3d_cam, pt2d_proj, ptr_d_proj_d_pt3d, ptr_J_cam);

        if (jacobians) {
            Eigen::Map<Eigen::Matrix<double, 2, TransformType::N, Eigen::RowMajor>> J_trans(jacobians[1]);
            J_trans.noalias() = d_proj_d_pt3d * d_pt3d_d_trans;
        }

        Eigen::Map<Vec2> res(residuals);
        res = pt2d_proj - pt2d_;

        return ret;
    }

private:
    const Vec3 pt3d_;
    const Vec2 pt2d_;
};

/********************** Reverse Projection **********************/

struct UnProjectCostFunctor {
public:
    using Vec2 = Eigen::Matrix<double, 2, 1>;
    using Vec3 = Eigen::Matrix<double, 3, 1>;

    UnProjectCostFunctor(const ChessBoard& board, const Vec2& pt_image, double pixel_val)
        : board_(board)
        , pt_image_(pt_image)
        , pixel_val_(pixel_val)
    {
    }

    template <unsigned R = 0>
    static double getPixVal(const double* const cam_params,
        const double* const trans_params,
        const ChessBoard& board, const Vec2& pt)
    {
        if constexpr (R == 0) {
            Vec3 pt_obj;
            if (CameraTransform::unproject(cam_params, trans_params, pt, pt_obj))
                return board.pixVal(pt_obj.x(), pt_obj.y());
            return -1.0;
        } else {
            using Kernel = GaussianKernel<R>;
            constexpr double space = 1.0 / Kernel::D;
            double val = 0.0;
            for (int dy = -R; dy <= int(R); ++dy)
                for (int dx = -R; dx <= int(R); ++dx) {
                    auto pv = getPixVal<0>(cam_params, trans_params, board,
                        pt + Vec2(dx * space, dy * space));
                    if (pv < 0.0) // unproject failed
                        return -1.0;
                    val += pv * Kernel::K[(dy + R) * Kernel::D + dx + R];
                }
            return val;
        }
    }

    bool operator()(const double* const cam_params,
        const double* const trans_params, double* residual) const
    {
        residual[0] = getPixVal<1>(cam_params, trans_params, board_, pt_image_) - pixel_val_;
        return true;
    }

private:
    template <unsigned _R>
    class GaussianKernel {
    public:
        static constexpr unsigned R = _R; //!< radius
        static constexpr unsigned D = R * 2 + 1; //!< diameter

        /// one-dimension kernel
        static constexpr auto k = []() -> std::array<double, D> {
            if constexpr (R == 0)
                return { 1.0 };
            else if constexpr (R == 1)
                return { 0.25, 0.5, 0.25 };
            else if constexpr (R == 2)
                return { 0.0625, 0.25, 0.375, 0.25, 0.0625 };
            else {
                static_assert(R < 3, "un-supported large window");
                return {};
            }
        }();

        /// kernel matrix
        static constexpr auto K = []() -> std::array<double, D * D> {
            std::array<double, D * D> ret {};
            for (size_t i = 0; i < D; ++i)
                for (size_t j = 0; j < D; ++j)
                    ret[i * D + j] = k[i] * k[j];
            return ret;
        }();
    };

private:
    const ChessBoard& board_;
    const Vec2 pt_image_;
    double pixel_val_;
};

} //namespace bxg
