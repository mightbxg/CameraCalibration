#pragma once

#include "camera/brown_camera.hpp"
#include "geometry/transform.hpp"
#include <ceres/ceres.h>

namespace bxg {
using CameraType = BrownCamera<double>;
using TransformType = RigidTransform<double>;

class PoseLocalParameterization : public ceres::LocalParameterization {
public:
    virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const override
    {
        using VecN = TransformType::VecN;
        using Vec3 = TransformType::Vec3;
        using Mat33 = TransformType::Mat33;
        Eigen::Map<const VecN> params(x);
        Eigen::Map<const VecN> del(delta);
        Eigen::Map<VecN> params_p_delta(x_plus_delta);

        Mat33 R = TransformType::toRotationMatrix(params.head<3>());
        Vec3 t = params.tail<3>();
        Mat33 dR = TransformType::toRotationMatrix(del.head<3>());
        Vec3 dt = del.tail<3>();
        params_p_delta.head<3>() = TransformType::toRotationVector(dR * R);
        params_p_delta.tail<3>() = dR * t + dt;
        return true;
    }
    virtual bool ComputeJacobian(const double* /*x*/, double* jacobian) const override
    {
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> jac(jacobian);
        jac.setIdentity();
        return true;
    }
    virtual int GlobalSize() const override { return 6; }
    virtual int LocalSize() const override { return 6; }
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
        using CamT = BrownCamera<T>;
        using TransT = RigidTransform<T>;
        auto camera = CamT(Eigen::Map<const typename CamT::VecN>(cam_params));
        auto trans = TransT(Eigen::Map<const typename TransT::VecN>(trans_params));

        auto pt3d_cam = trans.transform({ T(pt3d_[0]), T(pt3d_[1]), T(pt3d_[2]) });
        Eigen::Matrix<T, 2, 1> pt2d_proj;
        bool ret = camera.project(pt3d_cam, pt2d_proj);
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

} //namespace bxg
