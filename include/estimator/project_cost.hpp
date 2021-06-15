#pragma once

#include "camera/brown_camera.hpp"
#include "geometry/transform.hpp"
#include <ceres/ceres.h>

namespace bxg {
using CameraType = BrownCamera<double>;
using TransformType = RigidTransform<double>;

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

        auto pt3d_cam = trans.transform(pt3d_, jacobians[1]);
        Vec2 pt2d_proj;
        bool ret = camera.project(pt3d_cam, pt2d_proj, nullptr, jacobians[0]);

        Eigen::Map<Vec2> res(residuals);
        res = pt2d_proj - pt2d_;

        return ret;
    }

private:
    const Vec3 pt3d_;
    const Vec2 pt2d_;
};

} //namespace bxg
