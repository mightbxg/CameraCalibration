#pragma once

#include <Eigen/Core>
#include <vector>

namespace bxg {

enum class ReportType : uint8_t {
    NONE,
    BRIEF,
    FULL
};

class CameraCalibrator {
public:
    using Scalar = double;
    using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
    using CameraParams = std::vector<Scalar>;
    using TransformParams = Eigen::Matrix<Scalar, 7, 1>;
    struct SolveOption {
        bool minimizer_progress_to_stdout { false };
        ReportType report_type { ReportType::NONE };
    };

    Vec3 optimize(const std::vector<std::vector<Vec3>>& vpts3d,
        const std::vector<std::vector<Vec2>>& vpts2d, CameraParams& params,
        std::vector<Scalar>* covariance = nullptr);

public:
    SolveOption options;

private:
    // estimated params
    CameraParams cam_;
    std::vector<TransformParams> transforms_;

    // input data
};

} //namespace bxg
