#pragma once
#include <fstream>
#include <Eigen/Eigen>
#include <vector>
#include <deque>
#include <mutex>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/distances.h>
#include <pcl/common/eigen.h>
#include "ikd-Tree/ikd_Tree.h"

#include "LogTool.h"
#include "Timer.h"
#include "FileOperation.h"
#include "EigenMath.h"
#include "MathTools.h"

using namespace std;
using namespace Eigen;
using namespace EigenMath;

#define G_m_s2 (9.81) // Gravaty const in GuangDong/China

#define VEC_FROM_ARRAY(v) v[0], v[1], v[2]
#define MAT_FROM_ARRAY(v) v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]
#define LEFT_MULTIPLY_QUA(v) -v[1], -v[2], -v[3], \
                             v[0], -v[3], v[2],   \
                             v[3], v[0], -v[1],   \
                             -v[2], v[1], v[0];
#define CONSTRAIN(v, min, max) ((v > min) ? ((v < max) ? v : max) : min)
#define ARRAY_FROM_EIGEN(mat) mat.data(), mat.data() + mat.rows() * mat.cols()
#define STD_VEC_FROM_EIGEN(mat) vector<decltype(mat)::Scalar>(mat.data(), mat.data() + mat.rows() * mat.cols())
#define DEBUG_FILE_DIR(name) (string(string(ROOT_DIR) + "Log/" + name))
#define PCD_FILE_DIR(name) (string(string(ROOT_DIR) + "PCD/" + name))

/**
 * 6D位姿点云结构定义
 */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                  (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

using PointType = pcl::PointXYZINormal;
using PointCloudType = pcl::PointCloud<PointType>;
using PointVector = KD_TREE<PointType>::PointVector;

using V3D = Eigen::Vector3d;
using M3D = Eigen::Matrix3d;
using V3F = Eigen::Vector3f;
using M3F = Eigen::Matrix3f;
using QD = Eigen::Quaterniond;
using QF = Eigen::Quaternionf;

#define MD(a, b) Matrix<double, (a), (b)>
#define VD(a) Matrix<double, (a), 1>
#define MF(a, b) Matrix<float, (a), (b)>
#define VF(a) Matrix<float, (a), 1>

#define EYE3D (M3D::Identity())
#define EYE3F (M3F::Identity())
#define ZERO3D (V3D::Zero())
#define ZERO3F (V3F::Zero())
#define EYEQD (QD::Identity())
#define EYEQF (QF::Identity())

template <typename PointType>
float pointDistanceSquare(const PointType &p)
{
    return (p.x) * (p.x) + (p.y) * (p.y) + (p.z) * (p.z);
}

template <typename PointType>
float pointDistanceSquare(const PointType &p1, const PointType &p2)
{
    return pcl::squaredEuclideanDistance(p1, p2);
}

template <typename PointType>
float pointDistance(const PointType &p)
{
    return sqrt(pointDistanceSquare(p));
}

template <typename PointType>
float pointDistance(const PointType &p1, const PointType &p2)
{
    return sqrt(pointDistanceSquare(p1, p2));
}

inline const bool compare_timestamp(PointType &x, PointType &y) { return (x.curvature < y.curvature); };

inline bool check_for_not_converged(const double &timestamp, int step)
{
    static unsigned int cnt = 0;
    static double last_timestamp = 0;
    // LOG_WARN("check_for_not_converged = %f, %f, %f, %d", timestamp, last_timestamp, timestamp - last_timestamp, cnt);

    if (timestamp <= last_timestamp) // for test
        cnt = 0;

    if (cnt == 0)
    {
        last_timestamp = timestamp;
        ++cnt;
        return false;
    }

    bool flag = false;
    if (timestamp - last_timestamp > 60) // check only 60s
        return flag;

    if (cnt % step == 0)
    {
        if (timestamp - last_timestamp <= 1)
        {
            flag = true;
            // LOG_WARN("check_for_not_converged = %f, %d", timestamp - last_timestamp, cnt);
        }
        last_timestamp = timestamp;
    }
    ++cnt;
    return flag;
}

inline void check_time_interval(double &last_time, const double &cur_time, const double &expected_time, const std::string &what)
{
    auto delta_time = cur_time - last_time;
    if (last_time < 1e-6)
    {
        last_time = cur_time;
    }
    else if (delta_time >= 2.0 * expected_time)
    {
        LOG_WARN("%s time interval is %.3fs, more than expected %.3fs, last_time = %.3fs, cur_time = %.3fs.",
                 what.c_str(), delta_time, expected_time, last_time, cur_time);
    }
    last_time = cur_time;
}

inline bool check_imu_meas(const V3D &angular_velocity, const V3D &linear_acceleration, const vector<double> &imu_meas_check)
{
    if (std::abs(angular_velocity.x()) > imu_meas_check[0] || std::abs(angular_velocity.y()) > imu_meas_check[1] || std::abs(angular_velocity.z()) > imu_meas_check[2] ||
        std::abs(linear_acceleration.x()) > imu_meas_check[3] || std::abs(linear_acceleration.y()) > imu_meas_check[4] || std::abs(linear_acceleration.z()) > imu_meas_check[5])
    {
        LOG_WARN("imu_meas abnormal! ang_vel(%f, %f, %f), linear_acc(%f, %f, %f). Droped!",
                 angular_velocity.x(), angular_velocity.y(), angular_velocity.z(),
                 linear_acceleration.x(), linear_acceleration.y(), linear_acceleration.z());
        return false;
    }
    return true;
}

template <typename ikfom_state>
bool check_state_nan(const ikfom_state &state)
{
    if (state.pos.array().isNaN().any() || state.pos.array().isInf().any() ||
        state.rot.coeffs().array().isNaN().any() || state.rot.coeffs().array().isInf().any() ||
        state.vel.array().isNaN().any() || state.vel.array().isInf().any() ||
        state.ba.array().isNaN().any() || state.ba.array().isInf().any() ||
        state.bg.array().isNaN().any() || state.bg.array().isInf().any())
    {
        LOG_ERROR("state nan! pos(%f, %f, %f), rot(%f, %f, %f, %f), vel(%f, %f, %f), ba(%f, %f, %f), bg(%f, %f, %f). reset!",
                  state.pos.x(), state.pos.y(), state.pos.z(),
                  state.rot.x(), state.rot.y(), state.rot.z(), state.rot.w(),
                  state.vel.x(), state.vel.y(), state.vel.z(),
                  state.ba.x(), state.ba.y(), state.ba.z(),
                  state.bg.x(), state.bg.y(), state.bg.z());
        return true;
    }
    return false;
}

// #define DEDUB_MODE
