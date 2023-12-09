#include "ImuProcessor.h"

ImuProcessor::ImuProcessor()
    : b_first_frame_(true), imu_need_init_(true)
{
  init_iter_num = 1;
  mean_acc = V3D(0, 0, -1.0);
  mean_gyr = V3D(0, 0, 0);
  angvel_last = ZERO3D;
  last_imu_.reset(new ImuData());
}

ImuProcessor::~ImuProcessor() {}

void ImuProcessor::Reset()
{
  LOG_WARN("Reset ImuProcessor");
  mean_acc = V3D(0, 0, -1.0);
  mean_gyr = V3D(0, 0, 0);
  angvel_last = ZERO3D;
  imu_need_init_ = true;
  init_iter_num = 1;
  imu_states.clear();
  last_imu_.reset(new ImuData());
}

void ImuProcessor::set_imu_cov(const Eigen::Matrix<double, 12, 12> &imu_cov)
{
  Q = imu_cov;
}

void ImuProcessor::IMU_init(const MeasureCollection &meas, int &N)
{
  LOG_INFO("IMU Initializing: %.1f %%", double(N) / MAX_INI_COUNT * 100);

  if (b_first_frame_)
  {
    Reset();
    N = 1;
    b_first_frame_ = false;
    mean_acc = meas.imu.front()->linear_acceleration; // 加速度测量作为初始化均值
    mean_gyr = meas.imu.front()->angular_velocity;    // 角速度测量作为初始化均值
  }

  // 增量计算平均值和方差: https://blog.csdn.net/weixin_42040262/article/details/127345225
  for (const auto &imu : meas.imu)
  {
    const auto &cur_acc = imu->linear_acceleration;
    const auto &cur_gyr = imu->angular_velocity;

    mean_acc += (cur_acc - mean_acc) / N;
    mean_gyr += (cur_gyr - mean_gyr) / N;

    // cwiseProduct()对应系数相乘
    // 第一种是
    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);
    // 第二种是
    // cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - 上一次的mean_acc)  / N;
    // cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - 上一次的mean_gyr)  / N;

    N++;
  }
}

/**
 * 1.正向传播积分imu
 * 2.反向传播去畸变
 */
void ImuProcessor::UndistortPcl(const MeasureCollection &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudType &pcl_out)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu = meas.imu;                                 // 拿到当前的imu数据
  v_imu.push_front(last_imu_);                           // 将上一帧最后尾部的imu添加到当前帧头部的imu
  const double &imu_beg_time = v_imu.front()->timestamp; // 拿到当前帧头部的imu的时间（也就是上一帧尾部的imu时间戳）
  const double &imu_end_time = v_imu.back()->timestamp;  // 拿到当前帧尾部的imu的时间
  const double &pcl_beg_time = meas.lidar_beg_time;      // 当前帧pcl的开始时间
  const double &pcl_end_time = meas.lidar_end_time;      // 当前帧pcl的结束时间

  /*** sort point clouds by offset time ***/
  pcl_out = *(meas.lidar);

  /*** Initialize IMU pose ***/
  state_ikfom imu_state = kf_state.get_x(); // 获取上一次KF估计的后验状态作为本次IMU预测的初始状态
  imu_states.clear();
  // 将初始状态加入IMUpose中,包含有时间间隔，上一帧加速度，上一帧角速度，上一帧速度，上一帧位置，上一帧旋转矩阵
  imu_states.emplace_back(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix());

  /*** forward propagation at each imu point ***/
  V3D angvel_avr, acc_avr;
  double dt = 0;

  input_ikfom in;
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);

    if (tail->timestamp < last_lidar_end_time_)
      continue;

    // 中值积分
    angvel_avr = 0.5 * (head->angular_velocity + tail->angular_velocity);
    acc_avr = 0.5 * (head->linear_acceleration + tail->linear_acceleration);

    // 通过重力数值对加速度进行一下微调
    acc_avr = acc_avr * G_m_s2 / mean_acc.norm(); // - state_inout.ba;

    // 如果IMU开始时刻早于上次雷达最晚时刻(因为将上次最后一个IMU插入到此次开头了，所以会出现一次这种情况)
    if (head->timestamp < last_lidar_end_time_)
    {
      // 从上次雷达时刻末尾开始传播 计算与此次IMU结尾之间的时间差
      dt = tail->timestamp - last_lidar_end_time_;
      // dt = tail->timestamp - pcl_beg_time;
    }
    else
    {
      // 两个IMU时刻之间的时间间隔
      dt = tail->timestamp - head->timestamp;
    }

    in.acc = acc_avr;
    in.gyro = angvel_avr;
    kf_state.predict(dt, Q, in);

    /* save the poses at each IMU measurements */
    imu_state = kf_state.get_x();
    angvel_last = angvel_avr - imu_state.bg;                                    // 角速度 - 预测的角速度bias
    acc_s_last = imu_state.rot * (acc_avr - imu_state.ba) + imu_state.grav.vec; // 加速度 - 预测的加速度bias,并转到世界坐标系下
    double &&offs_t = tail->timestamp - pcl_beg_time;                           // 后一个IMU时刻距离此次雷达开始的时间间隔
    imu_states.emplace_back(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix());
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  // 把最后一帧IMU测量和当前帧最后一个点之间的也补上
  // 判断雷达结束时间是否晚于IMU，最后一个IMU时刻可能早于雷达末尾 也可能晚于雷达末尾
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  dt = note * (pcl_end_time - imu_end_time);
  kf_state.predict(dt, Q, in);

  imu_state = kf_state.get_x();        // 当前帧最后一个点状态
  last_imu_ = meas.imu.back();         // 保存最后一个IMU测量，以便于下一帧使用
  last_lidar_end_time_ = pcl_end_time; // 保存这一帧最后一个雷达测量的结束时间，以便于下一帧使用

  /*** undistort each lidar point (backward propagation) ***/
  if (pcl_out.points.begin() == pcl_out.points.end())
    return;
  auto it_pcl = pcl_out.points.end() - 1;
  for (auto it_kp = imu_states.end() - 1; it_kp != imu_states.begin(); it_kp--)
  {
    auto head = it_kp - 1;
    auto tail = it_kp;
    const M3D &R_imu = head->rot;
    const V3D &vel_imu = head->vel;
    const V3D &pos_imu = head->pos;
    const V3D &acc_imu = tail->acc;
    const V3D &gyr_imu = tail->gyr;

    // 点云时间需要迟于前一个IMU时刻，因为是在两个IMU时刻之间去畸变，此时默认雷达的时间戳在后一个IMU时刻之前
    for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--)
    {
      dt = it_pcl->curvature / double(1000) - head->offset_time;

      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
      /* 变换到“结束”帧，仅使用旋转
       * 注意: 补偿方向与帧的移动方向相反
       * 所以如果我们想补偿时间戳i到帧e的一个点
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei)  其中T_ei在全局框架中表示 */
      M3D R_i(R_imu * Exp(gyr_imu, dt));                                       // 当前点时刻的imu在世界坐标系下姿态
      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);                                   // 当前点位置(雷达坐标系下)
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos); // 当前点的时刻imu在世界坐标系下位置 - end时刻IMU在世界坐标系下的位置
      // 原理: 对应fastlio公式(10)，通过世界坐标系过度，将当前时刻IMU坐标系下坐标转换到end时刻IMU坐标系下
      // 去畸变补偿的公式推导:
      // conjugate(): 对于实数矩阵，共轭就是转置
      // imu_state.offset_R_L_I是雷达到imu的外参旋转矩阵 简单记为I^R_L
      // imu_state.offset_T_L_I是雷达到imu的外参平移向量 简单记为I^t_L
      // W^为世界坐标系下坐标，I^P为imu坐标系下点P坐标，L^P为雷达坐标系下坐标，e代表end时刻坐标
      // W^t_I为点所在时刻IMU在世界坐标系下的位置，W^t_I_e为end时刻IMU在世界坐标系下的位置
      // (1)世界坐标系下有: W^P = R_i * I^P + W^t_I = R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + W^t_I
      // (2)end时刻的世界坐标系下有: W^P = W^R_i_e * I^P_e + W^t_I_e, 其中: W^R_i_e = imu_state.rot, W^t_I_e = imu_state.pos
      // T_ei展开是pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos也就是点所在时刻IMU在世界坐标系下的位置 - end时刻IMU在世界坐标系下的位置 W^t_I-W^t_I_e
      // (3)联合(1)和(2)有: I^P_e = imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei)
      // (4)end时刻的imu坐标系下有: I^P_e = I^R_L * L^P_e + I^t_L, 其中: L^P_e就是P_compensate，即点在末尾时刻在雷达系的坐标
      // (5)联合(3)和(4)就有了下面的公式
      V3D P_compensate = imu_state.offset_R_L_I.conjugate() * (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I); // not accurate!

      // save Undistorted points and their rotation
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin())
        break;
    }
  }
}

void ImuProcessor::Process(const MeasureCollection &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudType::Ptr cur_pcl_un_)
{
  if (meas.imu.empty())
  {
    return;
  }
  assert(meas.lidar != nullptr);

  if (imu_need_init_)
  {
    /// The very first lidar frame
    IMU_init(meas, init_iter_num);

    last_imu_ = meas.imu.back(); // 将最后一帧的imu数据传入last_imu_中，暂时没用到

    if (init_iter_num > MAX_INI_COUNT)
    {
      LOG_INFO("IMU Initializing: %.1f %%", 100.0);
      imu_need_init_ = false;
      UndistortPcl(meas, kf_state, *cur_pcl_un_);
    }

    return;
  }
  if (!gravity_align_)
    gravity_align_ = true;

  // 正向传播积分imu, 反向传播去畸变
  UndistortPcl(meas, kf_state, *cur_pcl_un_);
}

void ImuProcessor::Process(const MeasureCollection &meas, PointCloudType::Ptr cur_pcl_un_, bool imu_en)
{
  if (imu_en)
  {
    if (meas.imu.empty())
      return;
    assert(meas.lidar != nullptr);

    if (imu_need_init_)
    {
      /// The very first lidar frame
      IMU_init(meas, init_iter_num);

      if (init_iter_num > MAX_INI_COUNT)
      {
        LOG_INFO("IMU Initializing: %.1f %%", 100.0);
        imu_need_init_ = false;
        *cur_pcl_un_ = *(meas.lidar);
      }
      return;
    }
    if (!gravity_align_)
      gravity_align_ = true;
    *cur_pcl_un_ = *(meas.lidar);
  }
  else
  {
    if (!b_first_frame_)
    {
      if (!gravity_align_)
        gravity_align_ = true;
    }
    else
    {
      b_first_frame_ = false;
      return;
    }
    *cur_pcl_un_ = *(meas.lidar);
  }
}

/**
 * @brief 将预设重力方向和测量到的重力方向对比，将imu初始姿态对齐到地图
 * @param preset_gravity 预设重力方向，也就是map方向
 * @param meas_gravity 测量到的重力
 * @param rot_init 返回的imu初始姿态
 */
void ImuProcessor::get_imu_init_rot(const V3D &preset_gravity, const V3D &meas_gravity, QD &rot_init)
{
  M3D hat_grav = hat(-preset_gravity);
  // sin(theta) = |a^b|/(|a|*|b|) = |axb|/(|a|*|b|)
  double align_sin = (hat_grav * meas_gravity).norm() / meas_gravity.norm() / preset_gravity.norm();
  // cos(theta) = a*b/(|a|*|b|)
  double align_cos = preset_gravity.transpose() * meas_gravity;
  align_cos = align_cos / preset_gravity.norm() / meas_gravity.norm();

  M3D rot_mat_init;

  if (align_sin < 1e-6)
  {
    if (align_cos > 1e-6)
      rot_mat_init = EYE3D;
    else
      rot_mat_init = -EYE3D;
  }
  else
  {
    // 沿着axb方向旋转对应夹角，得到imu初始姿态
    V3D align_angle = hat_grav * meas_gravity / (hat_grav * meas_gravity).norm() * acos(align_cos);
    rot_mat_init = Exp(align_angle);
  }

  V3D rpy = EigenMath::RotationMatrix2RPY(rot_mat_init);
  rpy.z() = 0;
  rot_init = EigenMath::RPY2Quaternion(rpy);
}
