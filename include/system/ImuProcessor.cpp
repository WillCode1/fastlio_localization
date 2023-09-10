#include "ImuProcessor.h"

const bool time_list(PointType &x, PointType &y) { return (x.curvature < y.curvature); };

ImuProcessor::ImuProcessor()
    : b_first_frame_(true), imu_need_init_(true)
{
  init_iter_num = 1;
  Q = process_noise_cov();
  cov_acc = V3D(0.1, 0.1, 0.1);
  cov_gyr = V3D(0.1, 0.1, 0.1);
  cov_bias_gyr = V3D(0.0001, 0.0001, 0.0001);
  cov_bias_acc = V3D(0.0001, 0.0001, 0.0001);
  mean_acc = V3D(0, 0, -1.0);
  mean_gyr = V3D(0, 0, 0);
  angvel_last = ZERO3D;
  last_imu_.reset(new ImuData());
}

ImuProcessor::~ImuProcessor() {}

void ImuProcessor::Reset()
{
  // LOG_WARN("Reset ImuProcessor");
  mean_acc = V3D(0, 0, -1.0);
  mean_gyr = V3D(0, 0, 0);
  angvel_last = ZERO3D;
  imu_need_init_ = true;
  init_iter_num = 1;
  imu_buffer.clear();
  imu_states.clear();
  last_imu_.reset(new ImuData());
}

void ImuProcessor::set_gyr_cov(const V3D &scaler)
{
  cov_gyr_scale = scaler;
}

void ImuProcessor::set_acc_cov(const V3D &scaler)
{
  cov_acc_scale = scaler;
}

void ImuProcessor::set_gyr_bias_cov(const V3D &b_g)
{
  cov_bias_gyr = b_g;
}

void ImuProcessor::set_acc_bias_cov(const V3D &b_a)
{
  cov_bias_acc = b_a;
}

/**
 * 1. 初始化重力、陀螺偏差、acc和陀螺仪协方差
 * 2. 将加速度测量值标准化为单位重力
 **/
void ImuProcessor::IMU_init(const MeasureCollection &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/

  V3D cur_acc, cur_gyr;

  // 这里应该是静止初始化
  if (b_first_frame_)
  {
    Reset();
    N = 1;
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;
    const auto &gyr_acc = meas.imu.front()->angular_velocity;
    mean_acc = imu_acc; // 加速度测量作为初始化均值
    mean_gyr = gyr_acc; // 角速度测量作为初始化均值
  }

  // 增量计算平均值和方差: https://blog.csdn.net/weixin_42040262/article/details/127345225
  for (const auto &imu : meas.imu)
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc = imu_acc;
    cur_gyr = gyr_acc;

    mean_acc += (cur_acc - mean_acc) / N;
    mean_gyr += (cur_gyr - mean_gyr) / N;

    // cwiseProduct()对应系数相乘
    // 第一种是
    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);
    // 第二种是
    // cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - 上一次的mean_acc)  / N;
    // cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - 上一次的mean_gyr)  / N;

    // cout<<"acc norm: "<<cur_acc.norm()<<" "<<mean_acc.norm()<<endl;

    N++;
  }
  state_ikfom init_state = kf_state.get_x();
  // init_state.grav = S2(-mean_acc / mean_acc.norm() * G_m_s2);

  // state_inout.rot = EYE3D; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
  init_state.bg = mean_gyr;                  // 静止初始化, 使用角速度测量作为陀螺仪偏差
  kf_state.change_x(init_state);

  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P();
  init_P.setIdentity();                                      // 将协方差矩阵置为单位阵
  init_P(6, 6) = init_P(7, 7) = init_P(8, 8) = 0.00001;      // 将协方差矩阵的位置和旋转的协方差置为0.00001
  init_P(9, 9) = init_P(10, 10) = init_P(11, 11) = 0.00001;  // 将协方差矩阵的速度和位姿的协方差置为0.00001
  init_P(15, 15) = init_P(16, 16) = init_P(17, 17) = 0.0001; // 将协方差矩阵的重力和姿态的协方差置为0.0001
  init_P(18, 18) = init_P(19, 19) = init_P(20, 20) = 0.001;  // 将协方差矩阵的陀螺仪偏差和姿态的协方差置为0.001
  init_P(21, 21) = init_P(22, 22) = 0.00001;                 // 将协方差矩阵的lidar和imu外参位移量的协方差置为0.00001
  kf_state.change_P(init_P);
  last_imu_ = meas.imu.back();                               // 将最后一帧的imu数据传入last_imu_中，暂时没用到
}

/**
 * 1.imu数据更新, 并积分imu位姿
 * 2.当前帧点云去畸变
 */
void ImuProcessor::UndistortPcl(const MeasureCollection &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudType &pcl_out)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu = meas.imu;                                            // 拿到当前的imu数据
  v_imu.push_front(last_imu_);                                      // 将上一帧最后尾部的imu添加到当前帧头部的imu
  const double &imu_beg_time = v_imu.front()->timestamp;            // 拿到当前帧头部的imu的时间（也就是上一帧尾部的imu时间戳）
  const double &imu_end_time = v_imu.back()->timestamp;             // 拿到当前帧尾部的imu的时间
  const double &pcl_beg_time = meas.lidar_beg_time;                 // 当前帧pcl的开始时间
  const double &pcl_end_time = meas.lidar_end_time;                 // 当前帧pcl的结束时间

  /*** sort point clouds by offset time ***/
  pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;

  /*** Initialize IMU pose ***/
  state_ikfom imu_state = kf_state.get_x(); // 获取上一次KF估计的后验状态作为本次IMU预测的初始状态
  imu_states.clear();
  // 将初始状态加入IMUpose中,包含有时间间隔，上一帧加速度，上一帧角速度，上一帧速度，上一帧位置，上一帧旋转矩阵
  imu_states.emplace_back(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix());

  /*** forward propagation at each imu point ***/
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu; // angvel_avr为平均角速度，acc_avr为平均加速度，acc_imu为imu加速度，vel_imu为imu速度，pos_imu为imu位置
  M3D R_imu;                                          // imu旋转矩阵

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

    // 原始测量的中值作为更新
    in.acc = acc_avr;
    in.gyro = angvel_avr;
    // 配置协方差矩阵
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
    // IMU前向传播，每次传播的时间间隔为dt
    kf_state.predict(dt, Q, in);

    /* save the poses at each IMU measurements */
    imu_state = kf_state.get_x();
    angvel_last = angvel_avr - imu_state.bg;               // 角速度 - 预测的角速度bias
    acc_s_last = imu_state.rot * (acc_avr - imu_state.ba); // 加速度 - 预测的加速度bias,并转到世界坐标系下
    for (int i = 0; i < 3; i++)
    {
      acc_s_last[i] += imu_state.grav[i]; // 加上重力得到世界坐标系的加速度: f_k = a^w - g
    }
    double &&offs_t = tail->timestamp - pcl_beg_time; // 后一个IMU时刻距离此次雷达开始的时间间隔
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
    R_imu = head->rot;
    vel_imu = head->vel;
    pos_imu = head->pos;
    acc_imu = tail->acc;
    angvel_avr = tail->gyr;

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
      M3D R_i(R_imu * SO3Math::Exp(angvel_avr * dt)); // 当前点时刻的imu在世界坐标系下姿态
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
    IMU_init(meas, kf_state, init_iter_num);

    last_imu_ = meas.imu.back();

    state_ikfom imu_state = kf_state.get_x();
    if (init_iter_num > MAX_INI_COUNT)
    {
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;

      cov_acc = cov_acc_scale;
      cov_gyr = cov_gyr_scale;
      // cov_acc = cov_acc.cwiseProduct(cov_acc_scale);
      // cov_gyr = cov_gyr.cwiseProduct(cov_gyr_scale);

      LOG_INFO("IMU Initial Done");
      // LOG_INFO("IMU Initial Done: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",\
      //          imu_state.grav[0], imu_state.grav[1], imu_state.grav[2], mean_acc.norm(), cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
    }

    return;
  }

  // 正向传播积分imu, 反向传播去畸变
  UndistortPcl(meas, kf_state, *cur_pcl_un_);
}
