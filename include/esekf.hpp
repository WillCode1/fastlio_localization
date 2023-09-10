#pragma once
#include <Eigen/Core>
#include "utility/Header.h"

namespace esekfom
{
struct state_esekf
{
    V3D pos = V3D::Zero();
    V3D vel = V3D::Zero();
    QD rot = QD::Identity();
    V3D ba = V3D::Zero();
    V3D bg = V3D::Zero();
    V3D grav = V3D::Zero();
    QD extrinsicR_LI = QD::Identity();
    V3D extrinsicT_LI = V3D::Zero();
};

struct imu_input
{
    V3D acc = V3D::Zero();
    V3D gyro = V3D::Zero();
};

struct process_noise
{
    V3D noise_g = V3D::Zero();
    V3D noise_a = V3D::Zero();
    V3D noise_bg = V3D::Zero();
    V3D noise_ba = V3D::Zero();
};

inline Eigen::MatrixXd process_noise_cov()
{
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(12, 12);
    cov.block<3, 3>(0, 0) = M3D::Identity() * 0.0001;
    cov.block<3, 3>(3, 3) = M3D::Identity() * 0.0001;
    cov.block<3, 3>(6, 6) = M3D::Identity() * 0.00001;
    cov.block<3, 3>(9, 9) = M3D::Identity() * 0.00001;
    return cov;
}

#define state_dim 24
#define process_noise_dim 12
class iesekf
{
public:
    using covariance = Eigen::Matrix<double , state_dim, state_dim>;
    using process_noise_covariance = Eigen::Matrix<double , process_noise_dim, process_noise_dim>;

    // state state;
    state_esekf state;
	covariance P = covariance::Zero();
    vector<double> epsilon = vector<double>(state_dim, 0.001);

    struct EffectFeature
    {
        V3D point_lidar;
        V3D norm_vec;
        double residual;
    };
    vector<bool> point_matched_surface;
    vector<PointVector> Nearest_Points;
    std::vector<EffectFeature> effect_features;

    void predict(const imu_input &imu_input, const process_noise_covariance &Q, const double &dt)
    {
        predict(state, imu_input, dt);
        auto Fx = get_Fx(state, imu_input, dt);
        auto Fw = get_Fw(state, imu_input, dt);
        P = Fx * P * Fx.transpose() + Fw * Q * Fw.transpose();
    }

    void iterate_update(const KD_TREE<PointType>::Ptr &ikdtree, const PointCloudType::Ptr &features)
    {
        point_matched_surface.resize(features->points.size());
        Nearest_Points.resize(features->points.size());

        bool is_converge = true;
        int converge_cnt = 0;
        double R = 0.001;
        int maximum_iter = 4;
        MatrixXd Hx, K;
        VectorXd z; // measure z
        Eigen::MatrixXd Jk_inv = Eigen::MatrixXd::Identity(state_dim, state_dim);
        auto state_iter_k = state;
        auto P_iter_k = P;
        for (int iter = 0; iter < maximum_iter; iter++)
        {
            if (!get_Hx_and_residual(ikdtree, features, Hx, z, is_converge))
                continue;

            VectorXd error_x = VectorXd::Zero(state_dim);
            error_x.block<3, 1>(0, 0) = SO3Math::Log(state.rot.toRotationMatrix().transpose() * state_iter_k.rot.toRotationMatrix());
            error_x.block<3, 1>(3, 0) = state_iter_k.pos - state.pos;
            error_x.block<3, 1>(6, 0) = state_iter_k.vel - state.vel;
            error_x.block<3, 1>(9, 0) = state_iter_k.bg - state.bg;
            error_x.block<3, 1>(12, 0) = state_iter_k.ba - state.ba;
            error_x.block<3, 1>(15, 0) = state_iter_k.grav - state.grav;
            error_x.block<3, 1>(18, 0) = SO3Math::Log(state.extrinsicR_LI.toRotationMatrix().transpose() * state_iter_k.extrinsicR_LI.toRotationMatrix());
            error_x.block<3, 1>(21, 0) = state_iter_k.extrinsicT_LI - state.extrinsicT_LI;

            const M3D& A_rot = SO3Math::J_l(error_x.block<3, 1>(0, 0));
            const M3D& A_extRLI = SO3Math::J_l(error_x.block<3, 1>(18, 0));
            Jk_inv.block<3, 3>(0, 0) = A_rot.transpose();
            Jk_inv.block<3, 3>(18, 0) = A_extRLI.transpose();
            P_iter_k = Jk_inv * P * Jk_inv.transpose();

            // if (true)
                K = (Hx.transpose() * Hx + (P_iter_k / R).inverse()).inverse() * Hx.transpose();
            // else
                // K = P_iter_k * Hx.transpose() * (Hx * P_iter_k * Hx.transpose() + R).inverse();
            // std::cout << "K: " << K << std::endl;

            error_x.block<3, 1>(0, 0) = A_rot.transpose() * error_x.block<3, 1>(0, 0);
            error_x.block<3, 1>(18, 0) = A_extRLI.transpose() * error_x.block<3, 1>(18, 0);
            auto update = -K * z - (Eigen::MatrixXd::Identity(state_dim, state_dim) - K * Hx) * error_x;
            // std::cout << update.transpose() << std::endl;

            state_iter_k.rot = state_iter_k.rot.toRotationMatrix() * SO3Math::Exp(update.block<3, 1>(0, 0));
            state_iter_k.rot.normalize();
            state_iter_k.pos = state_iter_k.pos + update.block<3, 1>(3, 0);
            state_iter_k.vel = state_iter_k.vel + update.block<3, 1>(6, 0);
            state_iter_k.bg = state_iter_k.bg + update.block<3, 1>(9, 0);
            state_iter_k.ba = state_iter_k.ba + update.block<3, 1>(12, 0);
            state_iter_k.grav = state_iter_k.grav + update.block<3, 1>(15, 0);
            state_iter_k.extrinsicR_LI = state_iter_k.extrinsicR_LI.toRotationMatrix() * SO3Math::Exp(update.block<3, 1>(18, 0));
            state_iter_k.extrinsicT_LI = state_iter_k.extrinsicT_LI + update.block<3, 1>(21, 0);

            is_converge = true;
            for (int i = 0; i < state_dim; i++)
            {
                if (update(i) > epsilon.at(i))
                {
                    is_converge = false;
                    break;
                }
            }

            if (is_converge)
                converge_cnt++;
            if(!converge_cnt && iter == maximum_iter - 2)
                is_converge = true;
            if(converge_cnt > 1)
                break;
        }
        state = state_iter_k;
        P = (Eigen::MatrixXd::Identity(state_dim, state_dim) - K * Hx) * P_iter_k;
    }

private:
    bool get_Hx_and_residual(const KD_TREE<PointType>::Ptr &ikdtree, const PointCloudType::Ptr &features,
                             MatrixXd &Hx, VectorXd &residual, bool is_converge)
    {
        // double match_start = omp_get_wtime();
        effect_features.clear();
        auto features_num = features->points.size();
        int NUM_MATCH_POINTS = 5;

        /** closest surface search and residual computation **/
#ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
        for (int i = 0; i < features_num; i++)
        {
            PointType point = features->points[i];

            /* transform to world frame */
            V3D p_lidar(point.x, point.y, point.z);
            V3D p_world = state.rot * (state.extrinsicR_LI * p_lidar + state.extrinsicT_LI) + state.pos;
            point.x = p_world.x();
            point.y = p_world.y();
            point.z = p_world.z(); 

            auto &points_near = Nearest_Points[i];

            if (is_converge)
            {
                /** Find the closest surfaces in the map **/
                vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
                ikdtree->Nearest_Search(point, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
                point_matched_surface[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false
                                                                                                                                      : true;
            }

            if (!point_matched_surface[i])
                continue;

            VF(4)
            abcd;
            if (esti_plane(abcd, points_near, 0.1f))
            {
                float dis = abcd(0) * point.x + abcd(1) * point.y + abcd(2) * point.z + abcd(3);
                float s = 1 - 0.9 * fabs(dis) / sqrt(p_lidar.norm());

                if (s > 0.9)
                {
                    EffectFeature effect_feat;
                    effect_feat.point_lidar = p_lidar;
                    effect_feat.norm_vec = V3D(abcd(0), abcd(1), abcd(2));
                    effect_feat.residual = dis;
                    effect_features.emplace_back(effect_feat);
                }
            }
        }

        if (effect_features.size() < 1)
        {
            // LOG_WARN("No Effective Points! \n");
            printf("No Effective Points! \n");
            return false;
        }

        // loger.match_time += omp_get_wtime() - match_start; // 返回从匹配开始时候所经过的时间
        // double solve_start = omp_get_wtime();              // 下面是solve求解的时间

        /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
        int effct_feat_num = effect_features.size();
        Hx = MatrixXd::Zero(effct_feat_num, state_dim);
        residual = VectorXd::Zero(effct_feat_num);

        // 求观测值与误差的雅克比矩阵，如论文式14以及式12、13
        for (int i = 0; i < effct_feat_num; i++)
        {
            const V3D& point_lidar = effect_features[i].point_lidar;
            const M3D& point_be_crossmat = SO3Math::get_skew_symmetric(point_lidar);
            const V3D& point_imu = state.extrinsicR_LI * point_lidar + state.extrinsicT_LI;
            const M3D& point_crossmat = SO3Math::get_skew_symmetric(point_imu);

            /*** get the normal vector of closest surface ***/
            const V3D &norm_vec = effect_features[i].norm_vec;

            /*** calculate the Measuremnt Jacobian matrix H ***/
            // 雅各比矩阵分子布局和分母布局的区别：(AR^Tu)^T = u^TR(A^T) = u^TR(-A)
            V3D C(state.rot.conjugate() * norm_vec);
            V3D A(point_crossmat * C);
            if (false)
            {
                V3D B(point_be_crossmat * state.extrinsicR_LI.conjugate() * C);
                // Hx.block<1, 12>(i, 0) << VEC_FROM_ARRAY(norm_vec), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
                Hx.block<1, 6>(i, 0) << VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(norm_vec);
                Hx.block<1, 6>(i, 18) << VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
            }
            else
            {
                // Hx.block<1, 12>(i, 0) << VEC_FROM_ARRAY(norm_vec), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
                Hx.block<1, 6>(i, 0) << VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(norm_vec);
            }

            /*** Measuremnt: distance to the closest plane ***/
            residual(i) = effect_features[i].residual;
        }
        // loger.solve_time += omp_get_wtime() - solve_start;
        return true;
    }

    // fast_lio2论文公式(2)
    void predict(state_esekf &state, const imu_input &imu_input, const double& dt)
    {
        // rot
        state.rot = state.rot * SO3Math::Exp((imu_input.gyro - state.bg) * dt);
        state.rot.normalize();
        // vel
        state.vel += (state.rot * (imu_input.acc - state.ba) + state.grav) * dt;
        // pos
        state.pos += state.vel * dt;
    }

    // eskf中，误差状态对各状态的雅克比矩阵F_x，对应fast_lio2论文公式(7)
    Eigen::MatrixXd get_Fx(state_esekf &state, const imu_input &imu_input, const double& dt)
    {
    	Eigen::MatrixXd Fx = Eigen::MatrixXd::Identity(state_dim, state_dim);
        auto I_33 = Eigen::Matrix3d::Identity();
        // rot
        Fx.block<3, 3>(0,  0) = SO3Math::Exp(-imu_input.gyro * dt);
        // Fx.block<3, 3>(0,  9) = -SO3Math::J_l(imu_input.gyro * dt).transpose() * dt;
        // Fx.block<3, 3>(0, 0) = I_33 + SO3Math::get_skew_symmetric(-imu_input.gyro * dt);
        Fx.block<3, 3>(0, 9) = -I_33 * dt;
        // pos
        Fx.block<3, 3>(3, 6) = I_33 * dt;
    	// vel
        Fx.block<3, 3>(6, 0) = -(state.rot * SO3Math::get_skew_symmetric(imu_input.acc * dt));
        Fx.block<3, 3>(6, 12) = -state.rot.toRotationMatrix() * dt;
        Fx.block<3, 3>(6, 15) = I_33 * dt;
    	return Fx;
    }

    // eskf中，误差状态对Noise的雅克比矩阵F_w，对应fast_lio2论文公式(7)
    Eigen::MatrixXd get_Fw(state_esekf &state, const imu_input &imu_input, const double &dt)
    {
        Eigen::MatrixXd Fw = Eigen::MatrixXd::Zero(state_dim, process_noise_dim);
        // Fw.block<3, 3>(0,  0) = -SO3Math::J_l(imu_input.gyro * dt).transpose() * dt;
        Fw.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity() * dt;  // gyro
        Fw.block<3, 3>(6, 3) = -state.rot.toRotationMatrix() * dt; // acc
        Fw.block<3, 3>(9, 6) = Eigen::Matrix3d::Identity() * dt;   // bg
        Fw.block<3, 3>(12, 9) = Eigen::Matrix3d::Identity() * dt;  // ba
        return Fw;
    }
};
// auto Qmat = Gmat_ * qmat_ * Gmat_.transpose();
}
