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
        const Scalar xw = p3d[0], yw = p3d[1], zw = p3d[2];

        Scalar x = xw / zw;
        Scalar y = yw / zw;
        Scalar r2 = x * x + y * y;
        Scalar s = Scalar(1) + r2 * (k1 + r2 * (k2 + k3 * r2));
        Scalar a1 = 2 * x * y;
        Scalar a2 = r2 + 2 * x * x;
        Scalar a3 = r2 + 2 * y * y;
        Scalar mx = s * x + a1 * p1 + a2 * p2;
        Scalar my = s * y + a1 * p2 + a3 * p1;

        proj[0] = fx * mx + cx;
        proj[1] = fy * my + cy;

        if (d_proj_d_p3d) {
            Scalar d_r2_d_x = 2 * x;
            Scalar d_r2_d_y = 2 * y;
            Scalar d_s_d_r2 = k1 + 2 * k2 * r2 + 3 * k3 * r2 * r2;

            Scalar d_mx_d_x = d_s_d_r2 * d_r2_d_x * x + s
                + 2 * y * p1 + d_r2_d_x * p2 + 4 * x * p2;
            Scalar d_mx_d_y = d_s_d_r2 * d_r2_d_y * x
                + 2 * x * p1 + d_r2_d_y * p2;

            Scalar d_my_d_x = d_s_d_r2 * d_r2_d_x * y
                + 2 * y * p2 + d_r2_d_x * p1;
            Scalar d_my_d_y = d_s_d_r2 * d_r2_d_y * y + s
                + 2 * x * p2 + d_r2_d_y * p1 + 4 * y * p1;

            Scalar zw_inv = Scalar(1) / zw;
            Scalar d_x_d_zw = -xw * zw_inv * zw_inv;
            Scalar d_y_d_zw = -yw * zw_inv * zw_inv;
            Scalar d_r2_d_zw = d_r2_d_x * d_x_d_zw + d_r2_d_y * d_y_d_zw;
            Scalar d_mx_d_zw = d_mx_d_x * d_x_d_zw + d_mx_d_y * d_y_d_zw + (p2 + x * d_s_d_r2) * d_r2_d_zw;
            if (0) {
                //Scalar d_xy_d_zw = -zw_inv * zw_inv * (xw * y + yw * x);
                d_mx_d_zw = d_s_d_r2 * d_r2_d_zw * x + s * d_x_d_zw
                    + 2 * p1 * (d_x_d_zw * y + d_y_d_zw * x)
                    + p2 * (d_r2_d_zw + 4 * x * d_x_d_zw);
            }
            {
                Scalar j1 = d_s_d_r2 * d_r2_d_zw * x + s * d_x_d_zw + 2 * p1 * (d_x_d_zw * y + d_y_d_zw * x) + p2 * d_r2_d_zw + p2 * 4 * x * d_x_d_zw
                    + (d_s_d_r2 * d_r2_d_x * x + d_r2_d_x * p2)
                        * d_x_d_zw
                    + (d_s_d_r2 * d_r2_d_y * x + d_r2_d_y * p2)
                        * d_y_d_zw;
                Scalar j2 = d_s_d_r2 * d_r2_d_zw * x + s * d_x_d_zw + 2 * p1 * (d_x_d_zw * y + d_y_d_zw * x) + p2 * d_r2_d_zw + p2 * 4 * x * d_x_d_zw;
            }
            Scalar d_my_d_zw = d_my_d_x * d_x_d_zw + d_my_d_y * d_y_d_zw
                + (p1 + y * d_s_d_r2) * (d_r2_d_x * d_x_d_zw + d_r2_d_y * d_y_d_zw);

            (*d_proj_d_p3d)(0, 0) = d_mx_d_x * zw_inv * fx;
            (*d_proj_d_p3d)(0, 1) = d_mx_d_y * zw_inv * fx;
            (*d_proj_d_p3d)(0, 2) = d_mx_d_zw * fx;
            (*d_proj_d_p3d)(1, 0) = d_my_d_x * zw_inv * fy;
            (*d_proj_d_p3d)(1, 1) = d_my_d_y * zw_inv * fy;
            (*d_proj_d_p3d)(1, 2) = d_my_d_zw * fy;
        }

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
