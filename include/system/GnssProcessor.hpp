#pragma once
#include <deque>
#include "DataDef.h"
#include "system/Header.h"
#include "global_localization/UtmCoordinate.h"
#include "global_localization/EnuCoordinate.h"
#define ENU

struct GnssPose
{
  GnssPose(const double &time = 0, const V3D &pos = ZERO3D, const QD &rot = EYEQD, const V3D &cov = ZERO3D)
      : timestamp(time), gnss_position(pos), gnss_quat(rot.normalized()), covariance(cov) {}

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double timestamp;
  V3D gnss_position;
  QD gnss_quat;
  V3D covariance;

  float current_gnss_interval;
  V3D lidar_pos_fix; // utm add extrinsic
};

/*
 * Adding gnss factors to the mapping requires a lower speed
 */
class GnssProcessor
{
public:
  GnssProcessor()
  {
    extrinsic_lidar2gnss.setIdentity();
#ifndef NO_LOGER
    file_pose_gnss = fopen(DEBUG_FILE_DIR("gnss_pose.txt").c_str(), "w");
    fprintf(file_pose_gnss, "# gnss trajectory\n# timestamp tx ty tz qx qy qz qw\n");
#endif
  }

  ~GnssProcessor()
  {
#ifndef NO_LOGER
    fclose(file_pose_gnss);
#endif
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  void set_extrinsic(const V3D &transl = ZERO3D, const M3D &rot = EYE3D);
  V3D gnss_global2local(const V3D &gnss_position);
  void gnss_handler(const GnssPose &gnss_raw);
  bool get_gnss_factor(GnssPose &thisGPS, const double &lidar_end_time, const double &odom_z);

  int numsv = 20;
  float rtk_age = 30;
  vector<float> gpsCovThreshold;

  float gnssValidInterval = 0.2;
  bool useGpsElevation = false;
  deque<GnssPose> gnss_buffer;

private:
  Eigen::Matrix4d extrinsic_lidar2gnss;
#ifndef NO_LOGER
  FILE *file_pose_gnss;
#endif
};

void GnssProcessor::set_extrinsic(const V3D &transl, const M3D &rot)
{
  extrinsic_lidar2gnss.setIdentity();
  extrinsic_lidar2gnss.topLeftCorner(3, 3) = rot;
  extrinsic_lidar2gnss.topRightCorner(3, 1) = transl;
}

V3D GnssProcessor::gnss_global2local(const V3D &gnss_position)
{
#ifdef ENU
  return enu_coordinate::Earth::LLH2ENU(gnss_position, true);
#else
  return utm_coordinate::LLAtoUTM2(gnss_position);
#endif
}

void GnssProcessor::gnss_handler(const GnssPose &gps_pose)
{
  gnss_buffer.push_back(gps_pose);
  // LogAnalysis::save_trajectory(file_pose_gnss, gps_pose.gnss_position, gps_pose.gnss_quat, gps_pose.timestamp);
}

bool GnssProcessor::get_gnss_factor(GnssPose &thisGPS, const double &lidar_end_time, const double &odom_z)
{
  while (!gnss_buffer.empty())
  {
    const auto &header_msg = gnss_buffer.front();
    if (header_msg.covariance(0) > gpsCovThreshold[0] || header_msg.covariance(1) > gpsCovThreshold[1] ||
        header_msg.covariance(2) > gpsCovThreshold[2] || header_msg.covariance(3) > gpsCovThreshold[3] ||
        header_msg.covariance(4) > gpsCovThreshold[4] || header_msg.covariance(5) > gpsCovThreshold[5])
    {
      LOG_WARN("GPS noise covariance is too large (%f, %f, %f, %f, %f, %f), ignored!",
               header_msg.covariance(0), header_msg.covariance(1),
               header_msg.covariance(2), header_msg.covariance(3),
               header_msg.covariance(4), header_msg.covariance(5));
      gnss_buffer.pop_front();
    }
    else if (header_msg.timestamp < lidar_end_time - gnssValidInterval)
    {
      gnss_buffer.pop_front();
    }
    else if (header_msg.timestamp > lidar_end_time + gnssValidInterval)
    {
      return false;
    }
    else
    {
      // find the one with the smallest time interval.
      thisGPS.current_gnss_interval = gnssValidInterval;
      while (!gnss_buffer.empty())
      {
        auto current_gnss_interval = std::abs(gnss_buffer.front().timestamp - lidar_end_time);
        if (current_gnss_interval < thisGPS.current_gnss_interval)
        {
          thisGPS = gnss_buffer.front();
          thisGPS.current_gnss_interval = current_gnss_interval;
          gnss_buffer.pop_front();
        }
        else
          break;
      }

      Eigen::Matrix4d gnss_pose = Eigen::Matrix4d::Identity();
      gnss_pose.topLeftCorner(3, 3) = thisGPS.gnss_quat.toRotationMatrix();
      gnss_pose.topRightCorner(3, 1) = thisGPS.gnss_position;
      gnss_pose *= extrinsic_lidar2gnss;
      thisGPS.lidar_pos_fix = gnss_pose.topRightCorner(3, 1);

      if (!useGpsElevation)
      {
        thisGPS.lidar_pos_fix(2) = odom_z;
        thisGPS.covariance(2) = 0.01;
      }
      return true;
    }
  }
  return false;
}
