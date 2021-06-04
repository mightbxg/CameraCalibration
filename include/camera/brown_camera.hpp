#pragma once

#include <Eigen/Core>

namespace bxg {

template <typename Scalar_ = double>
class BrownCamera {
public:
    static constexpr int N = 9; // k1,k2,k3,p1,p2
    using Scalar = Scalar_;

    using VecN = Eigen::Matrix<Scalar, N, 1>;
    using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;

    using Mat23 = Eigen::Matrix<Scalar, 2, 3>;
    using Mat32 = Eigen::Matrix<Scalar, 3, 2>;
    using Mat2N = Eigen::Matrix<Scalar, 2, N>;
    using Mat3N = Eigen::Matrix<Scalar, 3, N>;

    BrownCamera()
        : param_(VecN::Zero())
    {
    }
    explicit BrownCamera(const VecN& param)
        : param_(param)
    {
    }

    template <typename Scalar2>
    BrownCamera<Scalar2> cast() const
    {
        return BrownCamera<Scalar2>(param_.template cast<Scalar2>());
    }

    inline bool project(const Vec3& p3d, Vec2& proj, Mat23* d_proj_d_p3d = nullptr,
        Mat2N* d_proj_d_param = nullptr) const
    {
        const Scalar fx = param_[0];
        const Scalar fy = param_[1];
        const Scalar cx = param_[2];
        const Scalar cy = param_[3];
        const Scalar k1 = param_[4];
        const Scalar k2 = param_[5];
        const Scalar k3 = param_[6];
        const Scalar p1 = param_[7];
        const Scalar p2 = param_[8];

        Scalar x = p3d[0] / p3d[2];
        Scalar y = p3d[1] / p3d[2];
        Scalar r2 = x * x + y * y;
        Scalar s = Scalar(1) + r2 * (k1 + r2 * (k2 + k3 * r2));
        Scalar a1 = 2 * x * y;
        Scalar a2 = r2 + 2 * x * x;
        Scalar a3 = r2 + 2 * y * y;
        Scalar mx = s * x + a1 * p1 + a2 * p2;
        Scalar my = s * y + a1 * p2 + a3 * p1;

        proj[0] = fx * mx + cx;
        proj[1] = fy * my + cy;

        return p3d[2] > Eigen::NumTraits<Scalar>::dummy_precision();
    }

    /// @brief Projections used for unit-tests
    static std::vector<BrownCamera> getTestProjections()
    {
        std::vector<BrownCamera> res;
        VecN param;
        param << 631.6558837890625, 631.6558837890625, 637.0838623046875, 390.05694580078125,
            -0.03983112892232504, 0.03768857717889241, 0.0, 0.0007003028120895412, -0.0032084429231644357;
        res.emplace_back(param);
        return res;
    }

    /// @brief Resolutions used for unit-tests
    static std::vector<Eigen::Vector2i> getTestResolutions()
    {
        std::vector<Eigen::Vector2i> res;
        res.emplace_back(1280, 800);
        return res;
    }

    const VecN& param() const
    {
        return param_;
    }

private:
    VecN param_;
};

} //namespace bxg
