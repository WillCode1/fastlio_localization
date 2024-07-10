#ifndef USE_IKFOM_H
#define USE_IKFOM_H
#include <IKFoM_toolkit/esekfom/esekfom.hpp>

typedef MTK::vect<1, double> vect1;
typedef MTK::vect<2, double> vect2;
typedef MTK::vect<3, double> vect3;
typedef MTK::SO3<double> SO3;
typedef MTK::S2<double, 98090, 10000, 1> S2; 

MTK_BUILD_MANIFOLD(state_ikfom,
((vect3, pos))
((SO3, rot))
((SO3, offset_R_L_I))
((vect3, offset_T_L_I))
((vect3, vel))
((vect3, bg))
((vect3, ba))
((S2, grav))
);

MTK_BUILD_MANIFOLD(state_input,
((vect3, pos))
((SO3, rot))
((SO3, offset_R_L_I))
((vect3, offset_T_L_I))
((vect3, vel))
((vect3, bg))
((vect3, ba))
((vect3, gravity))
);

MTK_BUILD_MANIFOLD(state_output,
((vect3, pos))
((SO3, rot))
((SO3, offset_R_L_I))
((vect3, offset_T_L_I))
((vect3, vel))
((vect3, omg))
((vect3, acc))
((vect3, gravity))
((vect3, bg))
((vect3, ba))
);

MTK_BUILD_MANIFOLD(input_ikfom,
((vect3, acc))
((vect3, gyro))
);

MTK_BUILD_MANIFOLD(process_noise_input,
((vect3, ng))
((vect3, na))
((vect3, nbg))
((vect3, nba))
);

MTK_BUILD_MANIFOLD(process_noise_output,
((vect3, vel))
((vect3, ng))
((vect3, na))
((vect3, nbg))
((vect3, nba))
);

inline MTK::get_cov<process_noise_input>::type process_noise_cov(const double &gyr_cov, const double &acc_cov, const double &b_gyr_cov, const double &b_acc_cov)
{
	MTK::get_cov<process_noise_input>::type cov = MTK::get_cov<process_noise_input>::type::Zero();
	MTK::setDiagonal<process_noise_input, vect3, 0>(cov, &process_noise_input::ng, gyr_cov);	  // 0.03
	MTK::setDiagonal<process_noise_input, vect3, 3>(cov, &process_noise_input::na, acc_cov);	  // *dt 0.01 0.01 * dt * dt 0.05
	MTK::setDiagonal<process_noise_input, vect3, 6>(cov, &process_noise_input::nbg, b_gyr_cov);	  // *dt 0.00001 0.00001 * dt *dt 0.3 //0.001 0.0001 0.01
	MTK::setDiagonal<process_noise_input, vect3, 9>(cov, &process_noise_input::nba, b_acc_cov);	  // 0.001 0.05 0.0001/out 0.01
	return cov;
}

// fast_lio2论文公式(2), □+操作中增量对时间的雅克比矩阵
inline Eigen::Matrix<double, 24, 1> get_f(state_ikfom &s, const input_ikfom &in)
{
	// 这里的24个对应了pos(3), rot(3),offset_R_L_I(3),offset_T_L_I(3), vel(3), bg(3), ba(3), grav(3)
	Eigen::Matrix<double, 24, 1> res = Eigen::Matrix<double, 24, 1>::Zero();
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);
	res.block<3, 1>(0, 0) = s.vel;
	res.block<3, 1>(3, 0) = omega;
	res.block<3, 1>(12, 0) = s.rot.normalized() * (in.acc - s.ba) + s.grav.vec;
	return res;
}

// eskf中，误差状态对各状态的雅克比矩阵F_x中的去掉对角线的一部分(对应符合函数求导式展开后，g对误差状态的偏导数)，对应fast_lio2论文公式(7)
inline Eigen::Matrix<double, 24, 23> df_dx(state_ikfom &s, const input_ikfom &in)
{
	// 当中的23个对应了status的维度计算，为pos(3), rot(3),offset_R_L_I(3),offset_T_L_I(3), vel(3), bg(3), ba(3), grav(2);
	Eigen::Matrix<double, 24, 23> cov = Eigen::Matrix<double, 24, 23>::Zero();
	// pos
	cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
	// phi
	cov.template block<3, 3>(3, 15) = -Eigen::Matrix3d::Identity();
	// vel
	vect3 acc;
	in.acc.boxminus(acc, s.ba);
	cov.template block<3, 3>(12, 3) = -s.rot.toRotationMatrix() * MTK::hat(acc);
	cov.template block<3, 3>(12, 18) = -s.rot.toRotationMatrix();
	Eigen::Matrix<state_ikfom::scalar, 2, 1> vec = Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
	Eigen::Matrix<state_ikfom::scalar, 3, 2> grav_matrix;
	s.S2_Mx(grav_matrix, vec, 21);									// 将vec的2*1矩阵转为grav_matrix的3*2矩阵
	cov.template block<3, 2>(12, 21) = grav_matrix;
	return cov;
}

// eskf中，误差状态对Noise的雅克比矩阵F_w，对应fast_lio2论文公式(7)
inline Eigen::Matrix<double, 24, 12> df_dw(state_ikfom &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 24, 12> cov = Eigen::Matrix<double, 24, 12>::Zero();
	cov.template block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity(); // w
	cov.template block<3, 3>(12, 3) = -s.rot.toRotationMatrix();   // a
	cov.template block<3, 3>(15, 6) = Eigen::Matrix3d::Identity(); // bg
	cov.template block<3, 3>(18, 9) = Eigen::Matrix3d::Identity(); // ba
	return cov;
}

// point-lio
inline Eigen::Matrix<double, 24, 24> process_noise_cov_input(const double &gyr_cov_input, const double &acc_cov_input, const double &b_gyr_cov, const double &b_acc_cov)
{
	Eigen::Matrix<double, 24, 24> cov;
	cov.setZero();
	cov.block<3, 3>(3, 3).diagonal() << gyr_cov_input, gyr_cov_input, gyr_cov_input;
	cov.block<3, 3>(12, 12).diagonal() << acc_cov_input, acc_cov_input, acc_cov_input;
	cov.block<3, 3>(15, 15).diagonal() << b_gyr_cov, b_gyr_cov, b_gyr_cov;
	cov.block<3, 3>(18, 18).diagonal() << b_acc_cov, b_acc_cov, b_acc_cov;
	return cov;
}

inline Eigen::Matrix<double, 30, 30> process_noise_cov_output(const double &vel_cov, const double &gyr_cov_output, const double &acc_cov_output, const double &b_gyr_cov, const double &b_acc_cov)
{
	Eigen::Matrix<double, 30, 30> cov;
	cov.setZero();
	cov.block<3, 3>(12, 12).diagonal() << vel_cov, vel_cov, vel_cov;
	cov.block<3, 3>(15, 15).diagonal() << gyr_cov_output, gyr_cov_output, gyr_cov_output;
	cov.block<3, 3>(18, 18).diagonal() << acc_cov_output, acc_cov_output, acc_cov_output;
	cov.block<3, 3>(24, 24).diagonal() << b_gyr_cov, b_gyr_cov, b_gyr_cov;
	cov.block<3, 3>(27, 27).diagonal() << b_acc_cov, b_acc_cov, b_acc_cov;
	return cov;
}

inline Eigen::Matrix<double, 24, 1> get_f_input(state_input &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 24, 1> res = Eigen::Matrix<double, 24, 1>::Zero();
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);
	res.block<3, 1>(0, 0) = s.vel;
	res.block<3, 1>(3, 0) = omega;
	res.block<3, 1>(12, 0) = s.rot.normalized() * (in.acc - s.ba) + s.gravity;
	return res;
}

inline Eigen::Matrix<double, 30, 1> get_f_output(state_output &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 30, 1> res = Eigen::Matrix<double, 30, 1>::Zero();
	res.block<3, 1>(0, 0) = s.vel;
	res.block<3, 1>(3, 0) = s.omg;
	res.block<3, 1>(12, 0) = s.rot.normalized() * s.acc + s.gravity;
	return res;
}

inline Eigen::Matrix<double, 24, 24> df_dx_input(state_input &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 24, 24> cov = Eigen::Matrix<double, 24, 24>::Zero();
	// pos
	cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
	// phi
	cov.template block<3, 3>(3, 15) = -Eigen::Matrix3d::Identity();
	// vel
	vect3 acc;
	in.acc.boxminus(acc, s.ba);
	cov.template block<3, 3>(12, 3) = -s.rot.toRotationMatrix() * MTK::hat(acc);
	cov.template block<3, 3>(12, 18) = -s.rot.toRotationMatrix();
	cov.template block<3, 3>(12, 21) = Eigen::Matrix3d::Identity(); // grav_matrix;
	return cov;
}

inline Eigen::Matrix<double, 30, 30> df_dx_output(state_output &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 30, 30> cov = Eigen::Matrix<double, 30, 30>::Zero();
	// pos
	cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
	// phi
	cov.template block<3, 3>(3, 15) = Eigen::Matrix3d::Identity();
	// vel
	cov.template block<3, 3>(12, 3) = -s.rot.toRotationMatrix() * MTK::hat(s.acc);
	cov.template block<3, 3>(12, 18) = s.rot.toRotationMatrix();
	cov.template block<3, 3>(12, 21) = Eigen::Matrix3d::Identity(); // grav_matrix;
	return cov;
}

#endif