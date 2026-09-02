#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <unordered_set>
#include <deque>
#include "system/Header.h"
using namespace std;

enum LID_TYPE
{
  AVIA = 1,
  VELO16,
  OUST64
};

enum TIME_UNIT
{
  SEC = 0,
  MS = 1,
  US = 2,
  NS = 3
};

namespace velodyne_ros
{
// #define VEL_TIMESTAMP_TYPE float
#define VEL_TIMESTAMP_TYPE double
// #define VEL_TIMESTAMP_FIELD time
#define VEL_TIMESTAMP_FIELD timestamp
  struct EIGEN_ALIGN16 Point
  {
    PCL_ADD_POINT4D;
    float intensity;
    VEL_TIMESTAMP_TYPE VEL_TIMESTAMP_FIELD;
    uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
} // namespace velodyne_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (VEL_TIMESTAMP_TYPE, VEL_TIMESTAMP_FIELD, VEL_TIMESTAMP_FIELD)
    (std::uint16_t, ring, ring)
)

namespace ouster_ros
{
  struct EIGEN_ALIGN16 Point
  {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t ambient;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
} // namespace ouster_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    // use std::uint32_t to avoid conflicting with pcl::uint32_t
    (std::uint32_t, t, t)
    (std::uint16_t, reflectivity, reflectivity)
    (std::uint8_t, ring, ring)
    (std::uint16_t, ambient, ambient)
    (std::uint32_t, range, range)
)

class LidarProcessor
{
public:
  LidarProcessor();
  ~LidarProcessor();

  void init(int _n_scans,int _scan_rate, int _time_unit, double _blind, double detect_range);
  void avia_handler(const PointCloudType::Ptr &pl_orig, PointCloudType::Ptr &pcl_out);
  void oust64_handler(const pcl::PointCloud<ouster_ros::Point> &pl_orig, PointCloudType::Ptr &pcl_out);
  void velodyne_handler(const pcl::PointCloud<velodyne_ros::Point> &pl_orig, PointCloudType::Ptr &pcl_out);

  int scan_rate;
  double blind, detect_range;
  unordered_set<int> valid_ring;

private:
  int n_scans, time_unit;

  bool given_offset_time;
  float time_unit_scale;
  PointCloudType pl_surf;
};

LidarProcessor::LidarProcessor()
    : blind(0.01)
{
  n_scans = 6;
  scan_rate = 10;
  time_unit = US;

  blind = 0;
  detect_range = 1000;

  given_offset_time = false;
  time_unit_scale = 1.e-3f;
}

LidarProcessor::~LidarProcessor() {}

void LidarProcessor::init(int _n_scans, int _scan_rate, int _time_unit, double _blind, double _detect_range)
{
  n_scans = _n_scans;
  scan_rate = _scan_rate;
  time_unit = _time_unit;

  blind = _blind;
  detect_range = _detect_range;

  switch (time_unit)
  {
  case SEC:
    time_unit_scale = 1.e3f;
    break;
  case MS:
    time_unit_scale = 1.f;
    break;
  case US:
    time_unit_scale = 1.e-3f;
    break;
  case NS:
    time_unit_scale = 1.e-6f;
    break;
  default:
    time_unit_scale = 1.f;
    break;
  }
}

void LidarProcessor::avia_handler(const PointCloudType::Ptr &pl_orig, PointCloudType::Ptr &pcl_out)
{
  pl_surf.clear();
  int plsize = pl_orig->points.size();
  pl_surf.reserve(plsize);

  for (uint i = 1; i < plsize; i++)
  {
    double range = pointDistanceSquare(pl_orig->points[i]);
    if ((pointDistanceSquare(pl_orig->points[i], pl_orig->points[i - 1]) > 1e-7) &&
        (range > (blind * blind)) &&
        (range < (detect_range * detect_range)))
    {
      pl_surf.push_back(pl_orig->points[i]);
    }
  }

  *pcl_out = pl_surf;
}

void LidarProcessor::oust64_handler(const pcl::PointCloud<ouster_ros::Point> &pl_orig, PointCloudType::Ptr &pcl_out)
{
  pl_surf.clear();
  pl_surf.reserve(pl_orig.size());

  for (int i = 0; i < pl_orig.points.size(); i++)
  {
    double range = pointDistanceSquare(pl_orig.points[i]);

    if (range < (blind * blind) || range > (detect_range * detect_range))
      continue;

    Eigen::Vector3d pt_vec;
    PointType added_pt;
    added_pt.x = pl_orig.points[i].x;
    added_pt.y = pl_orig.points[i].y;
    added_pt.z = pl_orig.points[i].z;
    added_pt.intensity = pl_orig.points[i].intensity;
    added_pt.normal_x = 0;
    added_pt.normal_y = 0;
    added_pt.normal_z = 0;
    added_pt.curvature = (pl_orig.points[i].t - pl_orig.points[0].t) * time_unit_scale; // curvature unit: ms

    pl_surf.points.push_back(added_pt);
  }

  *pcl_out = pl_surf;
}

void LidarProcessor::velodyne_handler(const pcl::PointCloud<velodyne_ros::Point> &pl_orig, PointCloudType::Ptr &pcl_out)
{
  pl_surf.clear();
  int plsize = pl_orig.points.size();
  if (plsize == 0)
    return;
  pl_surf.reserve(plsize);

  /*** These variables only works when no point timestamps given ***/
  double omega_l = 0.361 * scan_rate; // scan angular velocity
  std::vector<bool> is_first(n_scans, true);
  std::vector<double> yaw_fp(n_scans, 0.0);   // yaw of first scan point
  std::vector<float> yaw_last(n_scans, 0.0);  // yaw of last scan point
  std::vector<float> time_last(n_scans, 0.0); // last offset time
  /*****************************************************************/

  if (pl_orig.points[plsize - 1].VEL_TIMESTAMP_FIELD > 0)
  {
    given_offset_time = true;
  }
  else
  {
    given_offset_time = false;
  }

  for (int i = 0; i < plsize; i++)
  {
    if (!valid_ring.empty() && valid_ring.count(pl_orig.points[i].ring) == 0)
    {
      continue;
    }

    PointType added_pt;
    // cout<<"!!!!!!"<<i<<" "<<plsize<<endl;

    added_pt.normal_x = 0;
    added_pt.normal_y = 0;
    added_pt.normal_z = 0;
    added_pt.x = pl_orig.points[i].x;
    added_pt.y = pl_orig.points[i].y;
    added_pt.z = pl_orig.points[i].z;
    added_pt.intensity = pl_orig.points[i].intensity;
    added_pt.curvature = (pl_orig.points[i].VEL_TIMESTAMP_FIELD - pl_orig.points[0].VEL_TIMESTAMP_FIELD) * time_unit_scale;

    if (!given_offset_time)
    {
      int layer = pl_orig.points[i].ring;
      double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

      if (is_first[layer])
      {
        // printf("layer: %d; is first: %d", layer, is_first[layer]);
        yaw_fp[layer] = yaw_angle;
        is_first[layer] = false;
        added_pt.curvature = 0.0;
        yaw_last[layer] = yaw_angle;
        time_last[layer] = added_pt.curvature;
        continue;
      }

      // compute offset time
      if (yaw_angle <= yaw_fp[layer])
      {
        added_pt.curvature = (yaw_fp[layer] - yaw_angle) / omega_l;
      }
      else
      {
        added_pt.curvature = (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
      }

      if (added_pt.curvature < time_last[layer])
        added_pt.curvature += 360.0 / omega_l;

      yaw_last[layer] = yaw_angle;
      time_last[layer] = added_pt.curvature;
    }

    double range = pointDistanceSquare(added_pt);

    if (range > (blind * blind) && range < (detect_range * detect_range))
    {
      pl_surf.points.push_back(added_pt);
    }
  }

  *pcl_out = pl_surf;
}
