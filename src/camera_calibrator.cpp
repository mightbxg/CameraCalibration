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
        Eigen::Matrix<double, 7, 1>& params) const
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
        if (success) {
            Eigen::Vector3d rvec(r[0], r[1], r[2]);
            Eigen::Quaterniond q(Eigen::AngleAxisd(rvec.norm(), rvec.normalized()));
            params << t[0], t[1], t[2], q.x(), q.y(), q.z(), q.w();
        }
        return success;
    }

private:
    Mat cam_mtx;
    Mat dis_cef;
};

} //anonymous namespace

namespace bxg {

CameraCalibrator::Vec3 CameraCalibrator::optimize(const vector<vector<Vec3>>& vpts3d,
    const vector<vector<Vec2>>& vpts2d, CameraParams& params, vector<Scalar>* covariance)
{
    using namespace ceres;
    ASSERT(vpts3d.size() == vpts2d.size());
    ASSERT(params.size() == CameraType::N);
    vector<TransformParams> transforms(vpts3d.size());

    PnPSolver pnp_solver(params);
    Problem::Options problem_options;
    problem_options.local_parameterization_ownership = DO_NOT_TAKE_OWNERSHIP;
    Problem problem(problem_options);
    double* ptr_cam_params = &params[0];
    LocalParameterization* local_parameterization = new PoseLocalParameterization();
    for (size_t frame_idx = 0; frame_idx < vpts3d.size(); ++frame_idx) {
        auto& trans_params = transforms[frame_idx];
        double* ptr_trans_params = &trans_params[0];
        const auto& pts3d = vpts3d[frame_idx];
        const auto& pts2d = vpts2d[frame_idx];
        ASSERT(pts3d.size() == pts2d.size());
        // get initial pose
        if (!pnp_solver.solve(pts3d, pts2d, trans_params))
            continue;
        for (size_t pt_idx = 0; pt_idx < pts3d.size(); ++pt_idx) {
            CostFunction* cost_func = new bxg::ProjectCostFunction(pts3d[pt_idx], pts2d[pt_idx]);
            problem.AddResidualBlock(cost_func, nullptr, { ptr_cam_params, ptr_trans_params });
            problem.SetParameterization(ptr_trans_params, local_parameterization);
        }
    }

    Solver::Options _options;
    _options.minimizer_progress_to_stdout = options.minimizer_progress_to_stdout;
    Solver::Summary summary;
    ceres::Solve(_options, &problem, &summary);
    switch (options.report_type) {
    case ReportType::BRIEF:
        cout << summary.BriefReport() << endl;
        break;
    case ReportType::FULL:
        cout << summary.FullReport() << endl;
        break;
    default:
        break;
    }

    if (covariance) {
        Covariance::Options cov_options;
        Covariance cov(cov_options);

        vector<pair<const double*, const double*>> cov_blocks;
        cov_blocks.push_back(make_pair(ptr_cam_params, ptr_cam_params));

        if (cov.Compute(cov_blocks, &problem)) {
            Eigen::Matrix<Scalar, CameraType::N, CameraType::N, Eigen::RowMajor> covs;
            cov.GetCovarianceBlock(ptr_cam_params, ptr_cam_params, covs.data());
            Eigen::Matrix<Scalar, CameraType::N, 1> diag = covs.diagonal();
            covariance->resize(CameraType::N);
            for (int i = 0; i < CameraType::N; ++i)
                covariance->at(i) = diag[i];
        }
    }

    vector<Scalar> residuals;
    Problem::EvaluateOptions eval_options;
    eval_options.apply_loss_function = false;
    eval_options.num_threads = 8;
    problem.Evaluate(eval_options, nullptr, &residuals, nullptr, nullptr);
    Vec3 errs(numeric_limits<Scalar>::max(), 0, 0); // min, max, avg
    for (size_t i = 0; i < residuals.size(); i += 2) {
        Scalar err = Vec2(residuals[i], residuals[i + 1]).norm();
        if (err < errs[0])
            errs[0] = err;
        else if (err > errs[1])
            errs[1] = err;
        errs[2] += err;
    }
    errs[2] /= residuals.size() / Scalar(2);

    cam_ = params;
    transforms_ = transforms;
    vpts3d_ = vpts3d;
    vpts2d_ = vpts2d;

    delete local_parameterization;
    return errs;
}

} //namespace bxg
