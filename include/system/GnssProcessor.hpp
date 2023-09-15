#pragma once
#include <deque>
#include "global_localization/UtmCoordinate.h"
#include "DataDef.h"
#include "utility/Header.h"

struct GnssPose
{
  GnssPose(const double &time = 0, const V3D &pos = ZERO3D, const V3D &cov = ZERO3D)
      : timestamp(time), gnss_position(pos), covariance(cov) {}

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double timestamp;
  V3D gnss_position;
  V3D rpy;
  V3D gnss_position_trans2imu; // utm add extrinsic
  V3D covariance;
  float current_gnss_interval;
};

/*
 * Adding gnss factors to the mapping requires a lower speed
 */
class GnssProcessor
{
public:
  GnssProcessor()
  {
    extrinsic_Lidar2Gnss.setIdentity();
    file_pose_gnss = fopen(DEBUG_FILE_DIR("gnss_pose.txt").c_str(), "w");
    fprintf(file_pose_gnss, "# gnss trajectory\n# timestamp tx ty tz qx qy qz qw\n");
  }

  ~GnssProcessor()
  {
    fclose(file_pose_gnss);
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  void set_extrinsic(const Eigen::Matrix4d &extrinsic = Eigen::Matrix4d::Identity());
  void gnss_handler(const GnssPose &gnss_raw);
  bool get_gnss_factor(GnssPose &thisGPS, const double &lidar_end_time, const double &odom_z);

  float gnssValidInterval = 0.2;
  float gpsCovThreshold = 2;
  bool useGpsElevation = false; // 是否使用gps高程优化
  deque<GnssPose> gnss_buffer;

private:
  bool check_mean_and_variance(const std::vector<V3D> &start_point, utm_coordinate::utm_point &utm_origin, const double &variance_thold);

private:
  Eigen::Matrix4d extrinsic_Lidar2Gnss;
  LogAnalysis loger;
  FILE *file_pose_gnss;
};

void GnssProcessor::set_extrinsic(const Eigen::Matrix4d &extrinsic)
{
  extrinsic_Lidar2Gnss = extrinsic;
}

bool GnssProcessor::check_mean_and_variance(const std::vector<V3D> &start_point, utm_coordinate::utm_point &utm_origin, const double &variance_thold)
{
  V3D mean = V3D::Zero();
  V3D variance = V3D::Zero();

  for (const V3D &vec : start_point)
  {
    mean += vec;
  }
  mean /= start_point.size();

  for (const V3D &vec : start_point)
  {
    V3D diff = vec - mean;
    variance.x() += diff.x() * diff.x();
    variance.y() += diff.y() * diff.y();
    variance.z() += diff.z() * diff.z();
  }
  variance /= (start_point.size() - 1); // 使用样本方差，除以 (n-1)

  LOG_WARN("check_mean_and_variance. mean = (%.5f, %.5f, %.5f), variance = (%.5f, %.5f, %.5f).", mean.x(), mean.y(), mean.z(), variance.x(), variance.y(), variance.z());

  if (variance.x() > variance_thold || variance.y() > variance_thold || variance.z() > variance_thold)
    return false;

  utm_origin.east = mean.x();
  utm_origin.north = mean.y();
  utm_origin.up = mean.z();
  return true;
}

void GnssProcessor::gnss_handler(const GnssPose &gnss_raw)
{
  static int count = 0;
  static utm_coordinate::utm_point utm_origin;
  static std::vector<V3D> start_point;
  count++;

  utm_coordinate::geographic_position lla;
  utm_coordinate::utm_point utm;
  lla.latitude = gnss_raw.gnss_position(0);
  lla.longitude = gnss_raw.gnss_position(1);
  lla.altitude = gnss_raw.gnss_position(2);
  utm_coordinate::LLAtoUTM(lla, utm);

  if (count <= 10)
  {
    utm_origin = utm;
    printf("--utm_origin: east: %.5f, north: %.5f, up: %.5f, zone: %s\n", utm_origin.east, utm_origin.north, utm_origin.up, utm_origin.zone.c_str());
    start_point.emplace_back(V3D(utm_origin.east, utm_origin.north, utm_origin.up));
    return;
  }
  else if (count == 11)
  {
    if (check_mean_and_variance(start_point, utm_origin, 0.05))
    {
      LOG_WARN("gnss init successfully! utm_origin_mean: east: %.5f, north: %.5f, up: %.5f, zone: %s.", utm_origin.east, utm_origin.north, utm_origin.up, utm_origin.zone.c_str());
    }
    else
    {
      count = 0;
      start_point.clear();
      LOG_WARN("gnss init failed!");
      return;
    }
  }

  GnssPose utm_pose = gnss_raw;
  utm_pose.gnss_position = V3D(utm.east - utm_origin.east, utm.north - utm_origin.north, utm.up - utm_origin.north);
  gnss_buffer.push_back(utm_pose);
  loger.save_gps_pose(file_pose_gnss, utm_pose.gnss_position, gnss_raw.rpy, gnss_raw.timestamp);
}

bool GnssProcessor::get_gnss_factor(GnssPose &thisGPS, const double &lidar_end_time, const double &odom_z)
{
  static PointType lastGPSPoint;
  while (!gnss_buffer.empty())
  {
    if (gnss_buffer.front().timestamp < lidar_end_time - gnssValidInterval)
    {
      gnss_buffer.pop_front();
    }
    else if (gnss_buffer.front().timestamp > lidar_end_time + gnssValidInterval)
    {
      return false;
    }
    else
    {
      // 找到时间间隔最小的
      thisGPS.current_gnss_interval = gnssValidInterval;
      auto current_gnss_interval = std::abs(gnss_buffer.front().timestamp - lidar_end_time);
      while (current_gnss_interval <= thisGPS.current_gnss_interval)
      {
        thisGPS.current_gnss_interval = current_gnss_interval;
        thisGPS = gnss_buffer.front();
        gnss_buffer.pop_front();
        current_gnss_interval = std::abs(gnss_buffer.front().timestamp - lidar_end_time);
      }

      // GPS噪声协方差太大，不能用
      float noise_x = thisGPS.covariance(0);
      float noise_y = thisGPS.covariance(1);
      if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
        continue;

      float gps_x = thisGPS.gnss_position(0);
      float gps_y = thisGPS.gnss_position(1);
      float gps_z = thisGPS.gnss_position(2);
      if (!useGpsElevation)
      {
        thisGPS.gnss_position(2) = odom_z;
        thisGPS.covariance(2) = 0.01;
      }

      // (0,0,0)无效数据
      if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
        continue;

      PointType curGPSPoint;
      curGPSPoint.x = gps_x;
      curGPSPoint.y = gps_y;
      curGPSPoint.z = gps_z;

      if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
        continue;
      else
        lastGPSPoint = curGPSPoint;

      Eigen::Matrix4d gnss_pose = EigenMath::CreateAffineMatrix(thisGPS.gnss_position, thisGPS.rpy);
      gnss_pose *= extrinsic_Lidar2Gnss;
      thisGPS.gnss_position_trans2imu = gnss_pose.topRightCorner(3, 1);
      return true;
    }
  }
  return false;
}
