#pragma once

#include "geometry/chessboard.hpp"
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
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

    static void balanceImage(const cv::Mat& src, cv::Mat& dst, const CameraParams& cam_param,
        const std::vector<TransformParams>& transforms, const ChessBoard& board = ChessBoard());

    /// optimize with forward projection (reproject error)
    Vec3 optimize(const std::vector<std::vector<Vec3>>& vpts3d,
        const std::vector<std::vector<Vec2>>& vpts2d, CameraParams& params,
        std::vector<Scalar>* covariance = nullptr, std::vector<TransformParams>* transforms = nullptr);

    /// optimize with backward projection (direct method)
    /// @note Must do forward projection optimization in advance
    void optimize(const cv::Mat& image, CameraParams& params,
        std::vector<Scalar>* covariance = nullptr, std::vector<TransformParams>* transforms = nullptr);

public:
    SolveOption options;

private:
    // estimated params
    CameraParams cam_;
    std::vector<TransformParams> transforms_;

    // input data
    std::vector<std::vector<Vec3>> vpts3d_;
    std::vector<std::vector<Vec2>> vpts2d_;
};

} //namespace bxg
