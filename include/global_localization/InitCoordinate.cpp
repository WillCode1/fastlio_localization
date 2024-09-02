#include "UtmCoordinate.h"
#include "EnuCoordinate.h"

namespace enu_coordinate
{
    Eigen::Vector3d Earth::_origin = Eigen::Vector3d::Zero();  // ECEF
    Eigen::Matrix3d Earth::_cne = Eigen::Matrix3d::Identity(); //
    bool Earth::_origin_setted = false;                        // 是否设置过圆心
}

namespace utm_coordinate
{
    bool origin_setted = false;
    utm_coordinate::utm_point utm_origin;
}
