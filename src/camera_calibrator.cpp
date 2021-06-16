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
    ASSERT(vpts3d.size() == vpts2d.size());
    ASSERT(params.size() == CameraType::N);

    Problem problem;

    return true;
}

} //namespace bxg
