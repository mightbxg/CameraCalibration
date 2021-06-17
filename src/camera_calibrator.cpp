#include "estimator/camera_calibrator.h"
#include "estimator/project_cost.hpp"
#include <opencv2/opencv.hpp>

using namespace std;

#define ASSERT(condition)                                                                            \
    {                                                                                                \
        if (!(condition))                                                                            \
            throw std::runtime_error(std::string("Assertion failed in file[") + __FILE__             \
                + "] line[" + std::to_string(__LINE__) + "] func[" + __func__ + "]: " + #condition); \
    }

namespace {
using namespace cv;
using namespace std;

class PnPSolver {
public:
    PnPSolver(const Eigen::Matrix<double, 9, 1>& p)
    {
        cam_mtx = (cv::Mat_<double>(3, 3) << p[0], 0.0, p[2],
            0.0, p[1], p[3], 0.0, 0.0, 1.0);
        dis_cef = (cv::Mat_<double>(5, 1) << p[4], p[5], p[7], p[8], p[6]);
    }
    PnPSolver(const vector<double>& p)
    {
        ASSERT(p.size() == 9);
        cam_mtx = (cv::Mat_<double>(3, 3) << p[0], 0.0, p[2],
            0.0, p[1], p[3], 0.0, 0.0, 1.0);
        dis_cef = (cv::Mat_<double>(5, 1) << p[4], p[5], p[7], p[8], p[6]);
    }
    bool solve(const vector<Eigen::Vector3d>& pts3d,
        const vector<Eigen::Vector2d>& pts2d,
        Eigen::Matrix<double, 6, 1>& params) const
    {
        vector<Point3d> _pts3d;
        vector<Point2d> _pts2d;
        Vec3d r, t;
        _pts3d.reserve(pts3d.size());
        for (const auto& p : pts3d)
            _pts3d.emplace_back(p.x(), p.y(), p.z());
        _pts2d.reserve(pts2d.size());
        for (const auto& p : pts2d)
            _pts2d.emplace_back(p.x(), p.y());
        bool success = cv::solvePnP(_pts3d, _pts2d, cam_mtx, dis_cef, r, t);
        params << r[0], r[1], r[2], t[0], t[1], t[2];
        return success;
    }

private:
    Mat cam_mtx;
    Mat dis_cef;
};

} //anonymous namespace

namespace bxg {

bool CameraCalibrator::optimize(const vector<vector<Vec3>>& vpts3d,
    const vector<vector<Vec2>>& vpts2d, Params& params)
{
    using namespace ceres;
    using TransformParams = ProjectCostFunction::TransformParams;
    ASSERT(vpts3d.size() == vpts2d.size());
    ASSERT(params.size() == CameraType::N);
    vector<TransformParams> transforms(vpts3d.size());

    PnPSolver pnp_solver(params);
    Problem problem;
    double* ptr_cam_params = &params[0];
    for (size_t frame_idx = 0; frame_idx < vpts3d.size(); ++frame_idx) {
        auto& trans_params = transforms[frame_idx];
        double* ptr_trans_params = &trans_params[0];
        const auto& pts3d = vpts3d[frame_idx];
        const auto& pts2d = vpts2d[frame_idx];
        ASSERT(pts3d.size() == pts2d.size());
        // get initial pose
        if (!pnp_solver.solve(pts3d, pts2d, trans_params))
            continue;
        trans_params.head<3>() *= 2;
        for (size_t pt_idx = 0; pt_idx < pts3d.size(); ++pt_idx) {
            CostFunction* cost_func = new bxg::ProjectCostFunction(pts3d[pt_idx], pts2d[pt_idx]);
            //CostFunction* cost_func = new AutoDiffCostFunction<bxg::ProjectCostFunctor, 2, 9, 6>(new bxg::ProjectCostFunctor(pts3d[pt_idx], pts2d[pt_idx]));
            //CostFunction* cost_func = new NumericDiffCostFunction<bxg::ProjectCostFunctor, ceres::CENTRAL, 2, 9, 6>(new bxg::ProjectCostFunctor(pts3d[pt_idx], pts2d[pt_idx]));

            problem.AddResidualBlock(cost_func, nullptr, { ptr_cam_params, ptr_trans_params });
        }
    }

    Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << summary.BriefReport() << endl;

    return true;
}

} //namespace bxg
