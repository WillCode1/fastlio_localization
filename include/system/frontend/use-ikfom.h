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

MTK::get_cov<process_noise_input>::type process_noise_cov(const double &gyr_cov, const double &acc_cov, const double &b_gyr_cov, const double &b_acc_cov);

// fast_lio2论文公式(2), □+操作中增量对时间的雅克比矩阵
Eigen::Matrix<double, 24, 1> get_f(state_ikfom &s, const input_ikfom &in);

// eskf中，误差状态对各状态的雅克比矩阵F_x中的去掉对角线的一部分(对应符合函数求导式展开后，g对误差状态的偏导数)，对应fast_lio2论文公式(7)
Eigen::Matrix<double, 24, 23> df_dx(state_ikfom &s, const input_ikfom &in);

// eskf中，误差状态对Noise的雅克比矩阵F_w，对应fast_lio2论文公式(7)
Eigen::Matrix<double, 24, 12> df_dw(state_ikfom &s, const input_ikfom &in);

// point-lio
Eigen::Matrix<double, 24, 24> process_noise_cov_input(const double &gyr_cov_input, const double &acc_cov_input, const double &b_gyr_cov, const double &b_acc_cov);

Eigen::Matrix<double, 30, 30> process_noise_cov_output(const double &vel_cov, const double &gyr_cov_output, const double &acc_cov_output, const double &b_gyr_cov, const double &b_acc_cov);

Eigen::Matrix<double, 24, 1> get_f_input(state_input &s, const input_ikfom &in);

Eigen::Matrix<double, 30, 1> get_f_output(state_output &s, const input_ikfom &in);

Eigen::Matrix<double, 24, 24> df_dx_input(state_input &s, const input_ikfom &in);

Eigen::Matrix<double, 30, 30> df_dx_output(state_output &s, const input_ikfom &in);

#endif