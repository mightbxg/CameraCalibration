#include "estimator/camera_calibrator.h"
#include "estimator/project_cost.hpp"
#include "utility/kdtree.hpp"

#include <TestFuncs/TicToc.hpp>

using namespace std;

#define USE_ANALYTICAL_DIFF 0

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
            params << q.x(), q.y(), q.z(), q.w(), t[0], t[1], t[2];
        }
        return success;
    }

private:
    Mat cam_mtx;
    Mat dis_cef;
};

void imshowNormal(const string& wn, const Mat& image)
{
    namedWindow(wn, WINDOW_NORMAL);
    imshow(wn, image);
}

} //anonymous namespace

namespace bxg {

void CameraCalibrator::balanceImage(const Mat& src, Mat& dst, const CameraParams& cam_params,
    const vector<TransformParams>& transforms, const ChessBoard& board)
{
    CV_Assert(src.type() == CV_8UC1);

    auto pickSrcColor = [&src](const Vec2& pt, int radius = 2, int threshold = 10) -> float {
        const int row_stride = src.step1();
        const uchar* c = &src.at<uchar>(pt.y(), pt.x());
        const uchar* ptr = c - row_stride * radius - radius;
        uchar c_val = *c;
        int sum = 0, cnt = 0;
        for (int y = -radius; y < radius; ++y, ptr += row_stride) {
            for (int x = -radius; x < radius; ++x, ++ptr) {
                if (abs(*ptr - c_val) < threshold) {
                    sum += *ptr;
                    ++cnt;
                }
            }
        }
        return cnt > 0 ? float(sum) / float(cnt) : float(c_val);
    };

    // square centers
    vector<bool> center_colors;
    auto centers = board.squareCenters(&center_colors);
    vector<Vec2> cts[2];
    vector<float> colors[2];
    for (const auto& trans : transforms) {
        for (size_t i = 0; i < centers.size(); ++i) {
            Vec2 pt_image;
            const Vec2& pt = centers[i];
            if (CameraTransform::project(cam_params.data(), trans.data(), Vec3(pt.x(), pt.y(), 0), pt_image)
                && pt_image.x() >= 0 && pt_image.x() < src.cols && pt_image.y() >= 0 && pt_image.y() < src.rows) {
                int color = center_colors[i];
                cts[color].push_back(pt_image);
                colors[color].push_back(pickSrcColor(pt_image));
            }
        }
    }
    // there must be at least 2 centers in the image
    CV_Assert(cts[0].size() > 1 && cts[1].size() > 1);
    using Kdt = KDTree<Vec2, 2>;
    Kdt kdt_color[2];
    kdt_color[0].build(cts[0]);
    kdt_color[1].build(cts[1]);

    // balence source image
    auto interpolateColor = [&](int x, int y, bool isWhite) -> float {
        const auto& kdt = kdt_color[isWhite];
        const auto& cls = colors[isWhite];
        vector<double> dists;
        auto ids = kdt.knnSearch(Vec2(x, y), 2, &dists);
        return (cls[ids[0]] * dists[1] + cls[ids[1]] * dists[0]) / (dists[0] + dists[1]);
    };
    dst = src.clone();
    dst.forEach<uchar>([&](uchar& pixel, const int pos[]) {
        constexpr uchar min_val = 50, max_val = 200;
        float min_val_real = interpolateColor(pos[1], pos[0], false);
        float max_val_real = interpolateColor(pos[1], pos[0], true);
        float alpha = (max_val - min_val) / (max_val_real - min_val_real);
        float beta = min_val - min_val_real * alpha;
        pixel = saturate_cast<uchar>(pixel * alpha + beta);
    });
}

CameraCalibrator::Vec3 CameraCalibrator::optimize(const vector<vector<Vec3>>& vpts3d,
    const vector<vector<Vec2>>& vpts2d, CameraParams& params, vector<Scalar>* covariance,
    vector<TransformParams>* transforms)
{
    using namespace ceres;
    ASSERT(vpts3d.size() == vpts2d.size());
    ASSERT(params.size() == CameraType::N);
    vector<TransformParams> _transforms(vpts3d.size());

    PnPSolver pnp_solver(params);
    Problem::Options problem_options;
    problem_options.local_parameterization_ownership = DO_NOT_TAKE_OWNERSHIP;
    Problem problem(problem_options);
    double* ptr_cam_params = &params[0];
#if USE_ANALYTICAL_DIFF
    LocalParameterization* local_parameterization = new PoseLocalParameterization();
#else
    LocalParameterization* local_parameterization = new Se3LocalParameterization();
#endif
    for (size_t frame_idx = 0; frame_idx < vpts3d.size(); ++frame_idx) {
        auto& trans_params = _transforms[frame_idx];
        double* ptr_trans_params = trans_params.data();
        const auto& pts3d = vpts3d[frame_idx];
        const auto& pts2d = vpts2d[frame_idx];
        ASSERT(pts3d.size() == pts2d.size());
        // get initial pose
        if (!pnp_solver.solve(pts3d, pts2d, trans_params))
            continue;
        for (size_t pt_idx = 0; pt_idx < pts3d.size(); ++pt_idx) {
#if USE_ANALYTICAL_DIFF
            CostFunction* cost_func = new bxg::ProjectCostFunction(pts3d[pt_idx], pts2d[pt_idx]);
#else
            CostFunction* cost_func = new AutoDiffCostFunction<bxg::ProjectCostFunctor,
                2, CameraType::N, Sophus::SE3d::num_parameters>(
                new bxg::ProjectCostFunctor(pts3d[pt_idx], pts2d[pt_idx]));
#endif
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

    if (transforms) {
        *transforms = _transforms;
    }

    if (1) { //test
        auto genBoardImage = [&](int idx) {
            dbg::TicToc::ScopedTimer st("genBoardImage");
            auto trans = _transforms[idx];
            auto board = ChessBoard();

            cv::Mat image = cv::Mat::zeros(120, 160, CV_8UC1);
            for (int r = 0; r < image.rows; ++r)
                for (int c = 0; c < image.cols; ++c) {
                    image.at<uchar>(r, c) = cvRound(UnProjectCostFunctor::getPixVal<1>(params.data(), trans.data(), board, c, r));
                }

            return image;
        };
        for (int i = 0; i < 4; ++i)
            imwrite(format("board_%d.png", i), genBoardImage(i));
    }

    cam_ = params;
    transforms_ = _transforms;
    vpts3d_ = vpts3d;
    vpts2d_ = vpts2d;

    delete local_parameterization;
    return errs;
}

} //namespace bxg
