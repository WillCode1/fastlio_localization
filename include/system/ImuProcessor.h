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
#include "DataDef.h"

/// *************Preconfiguration

#define MAX_INI_COUNT (10)

/// *************IMU Process and undistortion
class ImuProcessor
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcessor();
  ~ImuProcessor();

  void Reset();
  void set_gyr_cov(const V3D &scaler);
  void set_acc_cov(const V3D &scaler);
  void set_gyr_bias_cov(const V3D &b_g);
  void set_acc_bias_cov(const V3D &b_a);
  void Process(const MeasureCollection &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudType::Ptr pcl_un_);

  V3D cov_acc;
  V3D cov_gyr;
  V3D cov_acc_scale;
  V3D cov_gyr_scale;
  V3D cov_bias_gyr;
  V3D cov_bias_acc;
  deque<ImuData::Ptr> imu_buffer;

private:
  void IMU_init(const MeasureCollection &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);
  void UndistortPcl(const MeasureCollection &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudType &pcl_in_out);

  ImuData::Ptr last_imu_;
  vector<ImuState> imu_states;
  V3D mean_acc;
  V3D mean_gyr;
  V3D angvel_last;
  V3D acc_s_last;

  Eigen::Matrix<double, 12, 12> Q;

  double last_lidar_end_time_;
  int init_iter_num = 1;
  bool b_first_frame_ = true;
  bool imu_need_init_ = true;
};
