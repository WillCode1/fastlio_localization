#ifndef USE_IKFOM_H
#define USE_IKFOM_H
//#define USE_sparse

#include <IKFoM_toolkit/esekfom/esekfom.hpp>

typedef MTK::vect<3, double> vect3;
typedef MTK::SO3<double> SO3;
typedef MTK::S2<double, 98090, 10000, 1> S2; 
typedef MTK::vect<1, double> vect1;
typedef MTK::vect<2, double> vect2;

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

MTK_BUILD_MANIFOLD(input_ikfom,
((vect3, acc))
((vect3, gyro))
);

MTK_BUILD_MANIFOLD(process_noise_ikfom,
((vect3, ng))
((vect3, na))
((vect3, nbg))
((vect3, nba))
);

MTK::get_cov<process_noise_ikfom>::type process_noise_cov()
{
	MTK::get_cov<process_noise_ikfom>::type cov = MTK::get_cov<process_noise_ikfom>::type::Zero();
	MTK::setDiagonal<process_noise_ikfom, vect3, 0>(cov, &process_noise_ikfom::ng, 0.0001);	  // 0.03
	MTK::setDiagonal<process_noise_ikfom, vect3, 3>(cov, &process_noise_ikfom::na, 0.0001);	  // *dt 0.01 0.01 * dt * dt 0.05
	MTK::setDiagonal<process_noise_ikfom, vect3, 6>(cov, &process_noise_ikfom::nbg, 0.00001); // *dt 0.00001 0.00001 * dt *dt 0.3 //0.001 0.0001 0.01
	MTK::setDiagonal<process_noise_ikfom, vect3, 9>(cov, &process_noise_ikfom::nba, 0.00001); // 0.001 0.05 0.0001/out 0.01
	return cov;
}

// fast_lio2论文公式(2), □+操作中增量对时间的雅克比矩阵
Eigen::Matrix<double, 24, 1> get_f(state_ikfom &s, const input_ikfom &in)
{
	// 这里的24个对应了pos(3), rot(3),offset_R_L_I(3),offset_T_L_I(3), vel(3), bg(3), ba(3), grav(3)
	Eigen::Matrix<double, 24, 1> res = Eigen::Matrix<double, 24, 1>::Zero();
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);
	res.block<3, 1>(0, 0) = s.vel;
	res.block<3, 1>(3, 0) = omega;
	res.block<3, 1>(12, 0) = s.rot * (in.acc - s.ba) + s.grav.vec;
	return res;
}

// eskf中，误差状态对各状态的雅克比矩阵F_x中的一部分(对应符合函数求导式展开后，g对误差状态的偏导数)，对应fast_lio2论文公式(7)
Eigen::Matrix<double, 24, 23> df_dx(state_ikfom &s, const input_ikfom &in)
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
Eigen::Matrix<double, 24, 12> df_dw(state_ikfom &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 24, 12> cov = Eigen::Matrix<double, 24, 12>::Zero();
	cov.template block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity(); // w
	cov.template block<3, 3>(12, 3) = -s.rot.toRotationMatrix();   // a
	cov.template block<3, 3>(15, 6) = Eigen::Matrix3d::Identity(); // bg
	cov.template block<3, 3>(18, 9) = Eigen::Matrix3d::Identity(); // ba
	return cov;
}

#endif