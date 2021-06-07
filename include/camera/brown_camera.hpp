#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

namespace bxg {

/** @brief Brown camera model
 *  @note The FOV should be below 100 degree
 */
template <typename Scalar_ = double>
class BrownCamera {
public:
    static constexpr int N = 9; // k1,k2,k3,p1,p2
    using Scalar = Scalar_;

    using VecN = Eigen::Matrix<Scalar, N, 1>;
    using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;

    using Mat22 = Eigen::Matrix<Scalar, 2, 2>;
    using Mat23 = Eigen::Matrix<Scalar, 2, 3>;
    using Mat32 = Eigen::Matrix<Scalar, 3, 2>;
    using Mat2N = Eigen::Matrix<Scalar, 2, N>;
    using Mat3N = Eigen::Matrix<Scalar, 3, N>;

    /// tan(50/180*PI)
    static constexpr Scalar MAX_VIEW_ANGLE_TAN = 1.19175359259421;

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

    inline bool project(const Vec3& pt3d, Vec2& proj, Mat23* d_proj_d_pt3d = nullptr,
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
        const Scalar xw = pt3d[0], yw = pt3d[1], zw = pt3d[2];

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

        if (d_proj_d_pt3d) {
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

            (*d_proj_d_pt3d)(0, 0) = d_mx_d_x * zw_inv * fx;
            (*d_proj_d_pt3d)(0, 1) = d_mx_d_y * zw_inv * fx;
            (*d_proj_d_pt3d)(0, 2) = (d_mx_d_x * d_x_d_zw + d_mx_d_y * d_y_d_zw) * fx;
            (*d_proj_d_pt3d)(1, 0) = d_my_d_x * zw_inv * fy;
            (*d_proj_d_pt3d)(1, 1) = d_my_d_y * zw_inv * fy;
            (*d_proj_d_pt3d)(1, 2) = (d_my_d_x * d_x_d_zw + d_my_d_y * d_y_d_zw) * fy;
        }

        if (d_proj_d_param) {
            (*d_proj_d_param).setZero();
            (*d_proj_d_param)(0, 0) = mx;
            (*d_proj_d_param)(0, 2) = Scalar(1);
            (*d_proj_d_param)(1, 1) = my;
            (*d_proj_d_param)(1, 3) = Scalar(1);

            (*d_proj_d_param)(0, 4) = fx * x * r2;
            (*d_proj_d_param)(1, 4) = fy * y * r2;
            d_proj_d_param->col(5) = d_proj_d_param->col(4) * r2;
            d_proj_d_param->col(6) = d_proj_d_param->col(5) * r2;

            (*d_proj_d_param)(0, 7) = fx * 2 * x * y;
            (*d_proj_d_param)(0, 8) = fx * (r2 + 2 * x * x);
            (*d_proj_d_param)(1, 7) = fy * (r2 + 2 * y * y);
            (*d_proj_d_param)(1, 8) = fy * 2 * x * y;
        }

        return pt3d.z() > Eigen::NumTraits<Scalar>::dummy_precision()
            && std::abs(pt3d.x() / pt3d.z()) < MAX_VIEW_ANGLE_TAN
            && std::abs(pt3d.y() / pt3d.z()) < MAX_VIEW_ANGLE_TAN;
    }

    template <unsigned ITER>
    inline Vec2 solveXY(const Vec2& mxy, Mat22& J) const
    {
        const Scalar k1 = param_[4];
        const Scalar k2 = param_[5];
        const Scalar k3 = param_[6];
        const Scalar p1 = param_[7];
        const Scalar p2 = param_[8];

        Vec2 xy = mxy;
        Scalar& x = xy[0];
        Scalar& y = xy[1];
        for (unsigned i = 0; i < ITER; ++i) {
            Scalar r2 = x * x + y * y;
            Scalar s = Scalar(1) + r2 * (k1 + r2 * (k2 + k3 * r2));
            Scalar a1 = 2 * x * y;
            Scalar a2 = r2 + 2 * x * x;
            Scalar a3 = r2 + 2 * y * y;
            Scalar mx = s * x + a1 * p1 + a2 * p2;
            Scalar my = s * y + a1 * p2 + a3 * p1;
            Vec2 residual(mx - mxy[0], my - mxy[1]);

            Scalar d_r2_d_x = 2 * x;
            Scalar d_r2_d_y = 2 * y;
            Scalar d_s_d_r2 = k1 + 2 * k2 * r2 + 3 * k3 * r2 * r2;
            J(0, 0) = d_s_d_r2 * d_r2_d_x * x + s
                + 2 * y * p1 + d_r2_d_x * p2 + 4 * x * p2;
            J(0, 1) = d_s_d_r2 * d_r2_d_y * x
                + 2 * x * p1 + d_r2_d_y * p2;
            J(1, 0) = d_s_d_r2 * d_r2_d_x * y
                + 2 * y * p2 + d_r2_d_x * p1;
            J(1, 1) = d_s_d_r2 * d_r2_d_y * y + s
                + 2 * x * p2 + d_r2_d_y * p1 + 4 * y * p1;

            xy += (J.transpose() * J).ldlt().solve(-J.transpose() * residual);
        }
        return xy;
    }

    inline bool unproject(const Vec2& pt2d, Vec3& pt3d, Mat32* d_pt3d_d_pt2d = nullptr,
        Mat3N* d_pt3d_d_param = nullptr) const
    {
        const Scalar fx = param_[0];
        const Scalar fy = param_[1];
        const Scalar cx = param_[2];
        const Scalar cy = param_[3];
        const Scalar u = pt2d[0];
        const Scalar v = pt2d[1];

        const Scalar mx = (u - cx) / fx;
        const Scalar my = (v - cy) / fy;
        Mat22 J_mxy_xy;
        auto xy = solveXY<4>(Vec2(mx, my), J_mxy_xy);
        Scalar x = xy.x(), y = xy.y();
        pt3d = Vec3(x, y, 1.0);

        if (d_pt3d_d_pt2d || d_pt3d_d_param) {
            // [dx_dmx dx_dmy]
            // [dy_dmx dy_dmy]
            Mat22 J_xy_mxy = J_mxy_xy.inverse();
            Vec2 d_xy_du = J_xy_mxy.col(0) / fx;
            Vec2 d_xy_dv = J_xy_mxy.col(1) / fy;

            if (d_pt3d_d_pt2d) {
                d_pt3d_d_pt2d->setZero();
                d_pt3d_d_pt2d->col(0).template head<2>() = d_xy_du;
                d_pt3d_d_pt2d->col(1).template head<2>() = d_xy_dv;
            }

            if (d_pt3d_d_param) {
                d_pt3d_d_param->setZero();
                d_pt3d_d_param->col(0).template head<2>() = -d_xy_du * mx;
                d_pt3d_d_param->col(1).template head<2>() = -d_xy_dv * my;
                d_pt3d_d_param->col(2).template head<2>() = -d_xy_du;
                d_pt3d_d_param->col(3).template head<2>() = -d_xy_dv;

                Scalar r2 = x * x + y * y;
                Scalar a1 = 2 * x * y;
                Scalar a2 = r2 + 2 * x * x;
                Scalar a3 = r2 + 2 * y * y;
                d_pt3d_d_param->col(4).template head<2>() = J_xy_mxy * Vec2(-x * r2, -y * r2);
                d_pt3d_d_param->col(5).template head<2>() = d_pt3d_d_param->col(4).template head<2>() * r2;
                d_pt3d_d_param->col(6).template head<2>() = d_pt3d_d_param->col(5).template head<2>() * r2;
                d_pt3d_d_param->col(7).template head<2>() = J_xy_mxy * Vec2(-a1, -a3);
                d_pt3d_d_param->col(8).template head<2>() = J_xy_mxy * Vec2(-a2, -a1);
            }
        }

        return std::abs(pt3d.x()) < MAX_VIEW_ANGLE_TAN
            && std::abs(pt3d.y()) < MAX_VIEW_ANGLE_TAN;
    }

    /// @brief Projections used for unit-tests
    static std::vector<BrownCamera> getTestProjections()
    {
        std::vector<BrownCamera> res;
        VecN param;
        param << 631.6558837890625, 632.6558837890625, 637.0838623046875, 390.05694580078125,
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

    void operator+=(const VecN& vec) { param_ += vec; }

private:
    VecN param_;
};

} //namespace bxg
