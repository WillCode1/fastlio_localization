#pragma once
#include <cmath>
#include <deque>
#include <mutex>
#include <thread>
#include <Eigen/Eigen>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include "use-ikfom.hpp"
#include "../DataDef.h"

/// *************Preconfiguration

#define MAX_INI_COUNT (10)
// #define MAX_INI_COUNT (100)

/// *************IMU Process and undistortion
class ImuProcessor
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcessor();
  ~ImuProcessor();

  void Reset();
  void set_imu_cov(const Eigen::Matrix<double, 12, 12> &imu_cov);
  void Process(const MeasureCollection &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudType::Ptr pcl_un_);
  void Process(const MeasureCollection &meas, PointCloudType::Ptr pcl_un_, bool imu_en);
  void get_imu_init_rot(const V3D &preset_gravity, const V3D &meas_gravity, M3D &rot_init);

  V3D cov_acc;
  V3D cov_gyr;

  bool imu_need_init_ = true;
  bool gravity_align_ = false;

  V3D mean_acc;
  V3D mean_gyr;
  int imu_rate = 200;

private:
  void IMU_init(const MeasureCollection &meas, int &N);
  void UndistortPcl(const MeasureCollection &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudType &pcl_in_out);

  ImuData::Ptr last_imu_;
  vector<ImuState> imu_states;
  V3D angvel_last;
  V3D acc_s_last;

  Eigen::Matrix<double, 12, 12> Q;

  double last_lidar_end_time_;
  int init_iter_num = 1;
  bool b_first_frame_ = true;
};
