#include "estimator/camera_calibrator.h"
#include "estimator/project_cost.hpp"

using namespace std;

namespace bxg {

bool CameraCalibrator::optimize(const vector<Vec3>& pts3d,
    const vector<Vec2>& pts2d, Params& params)
{
    return true;
}

} //namespace bxg
