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

    static bool optimize(const std::vector<Vec3>& pts3d,
        const std::vector<Vec2>& pts2d, Params& params);
};

} //namespace bxg
