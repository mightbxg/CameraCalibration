#include "estimator/camera_calibrator.h"
#include "estimator/project_cost.hpp"

using namespace std;

#define ASSERT(condition)                                                                            \
    {                                                                                                \
        if (!(condition))                                                                            \
            throw std::runtime_error(std::string("Assertion failed in file[") + __FILE__             \
                + "] line[" + std::to_string(__LINE__) + "] func[" + __func__ + "]: " + #condition); \
    }

namespace bxg {

bool CameraCalibrator::optimize(const vector<vector<Vec3>>& vpts3d,
    const vector<vector<Vec2>>& vpts2d, Params& params)
{
    using namespace ceres;
    using TransformParams = ProjectCostFunction::TransformParams;
    ASSERT(vpts3d.size() == vpts2d.size());
    ASSERT(params.size() == CameraType::N);
    vector<TransformParams> transforms;
    transforms.resize(vpts3d.size(), TransformParams::Zero());

    Problem problem;
    double* ptr_cam_params = &params[0];
    for (size_t frame_idx = 0; frame_idx < vpts3d.size(); ++frame_idx) {
        double* ptr_trans_params = &transforms[frame_idx][0];
        const auto& pts3d = vpts3d[frame_idx];
        const auto& pts2d = vpts2d[frame_idx];
        ASSERT(pts3d.size() == pts2d.size());
        for (size_t pt_idx = 0; pt_idx < pts3d.size(); ++pt_idx) {
            CostFunction* cost_func = new bxg::ProjectCostFunction(pts3d[pt_idx], pts2d[pt_idx]);
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
