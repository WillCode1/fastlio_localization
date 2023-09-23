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
#include "SO3_Math.h"

using namespace std;
using namespace Eigen;

#define G_m_s2 (9.81) // Gravaty const in GuangDong/China

#define VEC_FROM_ARRAY(v) v[0], v[1], v[2]
#define MAT_FROM_ARRAY(v) v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]
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

template <typename PointType>
float pointDistanceSquare(const PointType& p)
{
    return (p.x) * (p.x) + (p.y) * (p.y) + (p.z) * (p.z);
}

template <typename PointType>
float pointDistanceSquare(const PointType& p1, const PointType& p2)
{
    return pcl::squaredEuclideanDistance(p1, p2);
}

template <typename PointType>
float pointDistance(const PointType& p)
{
    return sqrt(pointDistanceSquare(p));
}

template <typename PointType>
float pointDistance(const PointType& p1, const PointType& p2)
{
    return sqrt(pointDistanceSquare(p1, p2));
}

inline bool check_for_not_converged(const double &timestamp, int step)
{
    static unsigned int cnt = 0;
    static double last_timestamp = 0;
    // LOG_WARN("check_for_not_converged = %f, %f, %f, %d", timestamp, last_timestamp, timestamp - last_timestamp, cnt);

    if (timestamp <= last_timestamp)         // for test
        cnt = 0;

    if (cnt == 0)
    {
        last_timestamp = timestamp;
        ++cnt;
        return false;
    }

    bool flag = false;
    if (timestamp - last_timestamp > 60)    // check only 60s
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

inline void check_time_interval(double &last_time, const double &cur_time, const double &expected_time, const std::string& what)
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

// #define DEDUB_MODE
