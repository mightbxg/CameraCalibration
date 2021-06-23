#pragma once

#include <Eigen/Core>
#include <vector>

namespace bxg {

class CameraCalibrator {
public:
    using Scalar = double;
    using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
    using Params = std::vector<Scalar>;

    static Vec3 optimize(const std::vector<std::vector<Vec3>>& vpts3d,
        const std::vector<std::vector<Vec2>>& vpts2d, Params& params,
        std::vector<Scalar>* covariance = nullptr);
};

} //namespace bxg
