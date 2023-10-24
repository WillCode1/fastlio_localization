#pragma once
#include "FastlioOdometry.hpp"

/*
 * @paper https://onlinelibrary.wiley.com/doi/pdf/10.1002/aisy.202200459
 * @remark 由于pointlio将点云去畸变融入了更新过程，无法直接获得去畸变的稠密点云feats_undistort，只能拿到“去畸变”的稀疏点云feats_down_world
 */
class PointlioOdometry : public FastlioOdometry
{
public:
    PointlioOdometry() : FastlioOdometry()
    {
        frontend_type = Pointlio;
    }

    virtual ~PointlioOdometry() {}

    virtual void init_estimator()
    {
        auto lidar_meas_model_input = [&](state_input &a, esekfom::pointlio_datastruct<double> &b) { this->lidar_meas_model_input(a, b); };
        auto lidar_meas_model_output = [&](state_output &a, esekfom::pointlio_datastruct<double> &b) { this->lidar_meas_model_output(a, b); };
        auto imu_meas_model_output = [&](state_output &a, esekfom::pointlio_datastruct<double> &b) { this->imu_meas_model_output(a, b); };
        kf_input.init_dyn_share_modified(get_f_input, df_dx_input, lidar_meas_model_input);
        kf_output.init_dyn_share_modified_2h(get_f_output, df_dx_output, lidar_meas_model_output, imu_meas_model_output);

        Eigen::Matrix<double, 24, 24> P_init = MD(24, 24)::Identity() * 0.01;
        P_init.block<3, 3>(21, 21) = MD(3, 3)::Identity() * 0.0001;
        P_init.block<6, 6>(15, 15) = MD(6, 6)::Identity() * 0.001;
        P_init.block<6, 6>(6, 6) = MD(6, 6)::Identity() * 0.0001;
        kf_input.change_P(P_init);

        Eigen::Matrix<double, 30, 30> P_init_output = MD(30, 30)::Identity() * 0.01;
        P_init_output.block<3, 3>(21, 21) = MD(3, 3)::Identity() * 0.0001;
        P_init_output.block<6, 6>(6, 6) = MD(6, 6)::Identity() * 0.0001;
        P_init_output.block<6, 6>(24, 24) = MD(6, 6)::Identity() * 0.001;
        kf_output.change_P(P_init_output);
    }

    virtual void set_extrinsic(const V3D &transl, const M3D &rot)
    {
        if (use_imu_as_input)
        {
            state_in = kf_input.get_x();
            state_in.offset_R_L_I = rot;
            state_in.offset_T_L_I = transl;
            kf_input.change_x(state_in);
        }
        else
        {
            state_out = kf_output.get_x();
            state_out.offset_R_L_I = rot;
            state_out.offset_T_L_I = transl;
            kf_output.change_x(state_out);
        }

        offset_Tli = transl;
        offset_Rli = rot;
    }

    virtual void init_state(shared_ptr<ImuProcessor> &imu)
    {
        acc_norm = imu->mean_acc.norm();
        if (imu_en)
        {
            while (measures->lidar_beg_time > imu_next.timestamp)
            {
                imu_last = imu_next;
                imu_next = *(imu_buffer.front());
                imu_buffer.pop_front();
            }
            state_in.bg = state_out.bg = imu->mean_gyr; // 静止初始化, 使用角速度测量作为陀螺仪偏差
        }
        state_in.gravity << VEC_FROM_ARRAY(gravity_init);
        state_out.gravity << VEC_FROM_ARRAY(gravity_init);
        state_out.acc = -state_out.gravity;

        kf_input.change_x(state_in);
        kf_output.change_x(state_out);
    }

    virtual void reset_state(const Eigen::Matrix4d &imu_pose)
    {
        Eigen::Quaterniond fine_tune_quat(M3D(imu_pose.topLeftCorner(3, 3)));
        if (use_imu_as_input)
        {
            state_in = kf_input.get_x();
            state_in.vel.setZero();
            state_in.ba.setZero();
            state_in.bg.setZero();
            state_in.offset_R_L_I = offset_Rli;
            state_in.offset_T_L_I = offset_Tli;
            state_in.gravity << VEC_FROM_ARRAY(gravity_init);
            state_in.pos = V3D(imu_pose.topRightCorner(3, 1));
            state_in.rot.coeffs() = Vector4d(fine_tune_quat.x(), fine_tune_quat.y(), fine_tune_quat.z(), fine_tune_quat.w());
            kf_input.change_x(state_in);
        }
        else
        {
            state_out = kf_output.get_x();
            state_out.vel.setZero();
            state_out.ba.setZero();
            state_out.bg.setZero();
            state_out.offset_R_L_I = offset_Rli;
            state_out.offset_T_L_I = offset_Tli;
            state_out.gravity << VEC_FROM_ARRAY(gravity_init);
            state_out.pos = V3D(imu_pose.topRightCorner(3, 1));
            state_out.rot.coeffs() = Vector4d(fine_tune_quat.x(), fine_tune_quat.y(), fine_tune_quat.z(), fine_tune_quat.w());
            kf_output.change_x(state_out);
        }
    }

    virtual bool sync_sensor_data()
    {
        static bool lidar_pushed = false;
        static double lidar_mean_scantime = 0.0;
        static int scan_num = 0;

        std::lock_guard<std::mutex> lock(mtx_buffer);
        if (lidar_buffer.empty() || imu_en && imu_buffer.empty())
        {
            return false;
        }

        /*** push a lidar scan ***/
        if (!lidar_pushed)
        {
            measures->lidar = lidar_buffer.front();
            measures->lidar_beg_time = time_buffer.front();
            if (measures->lidar->points.size() <= 1)
            {
                LOG_WARN("Too few input point cloud!\n");
                lidar_buffer.pop_front();
                time_buffer.pop_front();
                return false;
            }
            auto last_point_timestamp = measures->lidar->points.back().curvature;
            if (last_point_timestamp < 0.5 * lidar_mean_scantime)
            {
                lidar_end_time = measures->lidar_beg_time + lidar_mean_scantime / double(1000);
            }
            else
            {
                lidar_end_time = measures->lidar_beg_time + last_point_timestamp / double(1000);
                if (scan_num < INT_MAX)
                {
                    scan_num++;
                    lidar_mean_scantime += (last_point_timestamp - lidar_mean_scantime) / scan_num;
                }
            }

            measures->lidar_end_time = lidar_end_time;
            lidar_pushed = true;
        }

        if (!imu_en)
        {
            lidar_pushed = false;
            lidar_buffer.pop_front();
            time_buffer.pop_front();
            return true;
        }
        else if (latest_timestamp_imu < lidar_end_time)
        {
            return false;
        }

        /*** push imu data, and pop from imu buffer ***/
        if (imu->imu_need_init_)
        {
            double imu_time = imu_buffer.front()->timestamp;
            measures->imu.shrink_to_fit();
            while ((!imu_buffer.empty()) && (imu_time <= lidar_end_time))
            {
                measures->imu.emplace_back(imu_buffer.front());
                imu_last = imu_next;
                imu_next = *(imu_buffer.front());
                imu_buffer.pop_front();
                imu_time = imu_buffer.front()->timestamp;
            }
        }

        lidar_buffer.pop_front();
        time_buffer.pop_front();
        lidar_pushed = false;
        return true;
    }

    virtual bool run(PointCloudType::Ptr &feats_undistort)
    {
        if (loger.runtime_log && !loger.inited_first_lidar_beg_time)
        {
            loger.first_lidar_beg_time = measures->lidar_beg_time;
            loger.inited_first_lidar_beg_time = true;
        }
        loger.resetTimer();
        imu->Process(*measures, feats_undistort, imu_en);

        if (feats_undistort->empty() || (feats_undistort == NULL))
        {
            LOG_WARN("Wait for the imu to initialize completed!");
            return true;
        }

        if (!imu->gravity_align_)
            init_state(imu);

        loger.imu_process_time = loger.timer.elapsedLast();
        loger.feats_undistort_size = feats_undistort->points.size();
        loger.kdtree_size = ikdtree.size();
        if (use_imu_as_input)
            loger.dump_state_to_log(loger.fout_predict, state_in, measures->lidar_beg_time - loger.first_lidar_beg_time);
        else
            loger.dump_state_to_log(loger.fout_predict, state_out, measures->lidar_beg_time - loger.first_lidar_beg_time);

        /*** interval sample and downsample the feature points in a scan ***/
        feats_down_lidar->clear();
        for (int i = 0; i < feats_undistort->size(); i++)
            if (i % point_skip_num == 0)
            {
                feats_down_lidar->points.push_back(feats_undistort->points[i]);
            }
        if (space_down_sample)
        {
            surf_frame_ds_filter.setLeafSize(surf_frame_ds_res, surf_frame_ds_res, surf_frame_ds_res);
            surf_frame_ds_filter.setInputCloud(feats_down_lidar);
            surf_frame_ds_filter.filter(*feats_down_lidar);
        }
        sort(feats_down_lidar->points.begin(), feats_down_lidar->points.end(), compare_timestamp);
        time_seq = time_compressing(feats_down_lidar);

        feats_down_size = feats_down_lidar->points.size();
        loger.feats_down_size = feats_down_size;
        loger.downsample_time = loger.timer.elapsedLast();

        /*** iterated state estimation ***/
        feats_down_world->resize(feats_down_size);
        point_matched_surface.resize(feats_down_size);
        nearest_points.resize(feats_down_size);
        normvec->resize(feats_down_size);
        crossmat_list.reserve(feats_down_size);

        for (size_t i = 0; i < feats_down_size; ++i)
        {
            V3D point_this(feats_down_lidar->points[i].x, feats_down_lidar->points[i].y, feats_down_lidar->points[i].z);
            if (use_imu_as_input)
                pointLidarToWorld(point_this, point_this, kf_input.x_);
            else
                pointLidarToWorld(point_this, point_this, kf_output.x_);
            crossmat_list[i] = SO3Math::get_skew_symmetric(point_this);
        }

        if (use_imu_as_input)
        {
            double pcl_beg_time = measures->lidar_beg_time;
            idx = -1;
            for (k = 0; k < time_seq.size(); k++)
            {
                PointType &point_body = feats_down_lidar->points[idx + time_seq[k]];
                auto time_current = point_body.curvature / 1000.0 + pcl_beg_time;
                if (is_first_frame)
                {
                    while (time_current > imu_next.timestamp)
                    {
                        imu_last = imu_next;
                        imu_next = *(imu_buffer.front());
                        imu_buffer.pop_front();
                    }

                    is_first_frame = false;
                    time_predict_last = time_current;
                    time_update_last = time_current;
                    // if(prop_at_freq_of_imu)
                    {
                        input_in.gyro = imu_last.angular_velocity;
                        input_in.acc = imu_last.linear_acceleration;
                        input_in.acc = input_in.acc * G_m_s2 / acc_norm;
                    }
                }

                while (time_current > imu_next.timestamp) // && !imu_buffer.empty())
                {
                    imu_last = imu_next;
                    imu_next = *(imu_buffer.front());
                    imu_buffer.pop_front();
                    input_in.gyro = imu_last.angular_velocity;
                    input_in.acc = imu_last.linear_acceleration;
                    input_in.acc = input_in.acc * G_m_s2 / acc_norm;
                    double dt = imu_last.timestamp - time_predict_last;

                    // if(!prop_at_freq_of_imu)
                    // {
                    double dt_cov = imu_last.timestamp - time_update_last;
                    if (dt_cov > 0.0)
                    {
                        kf_input.predict(dt_cov, Q_input, input_in, false, true);
                        time_update_last = imu_last.timestamp; // time_current;
                    }
                    kf_input.predict(dt, Q_input, input_in, true, false);
                    time_predict_last = imu_last.timestamp;
                }

                double dt = time_current - time_predict_last;
                time_predict_last = time_current;
                // double propag_start = omp_get_wtime();

                if (!prop_at_freq_of_imu)
                {
                    double dt_cov = time_current - time_update_last;
                    if (dt_cov > 0.0)
                    {
                        kf_input.predict(dt_cov, Q_input, input_in, false, true);
                        time_update_last = time_current;
                    }
                }
                kf_input.predict(dt, Q_input, input_in, true, false);

                // propag_time += omp_get_wtime() - propag_start;
                // double t_update_start = omp_get_wtime();
                if (!kf_input.update_iterated_pointlio())
                {
                    idx += time_seq[k];
                    continue;
                }

                // solve_start = omp_get_wtime();
                // solve_time += omp_get_wtime() - solve_start;
                // update_time += omp_get_wtime() - t_update_start;

                for (int j = 0; j < time_seq[k]; j++)
                {
                    PointType &point_lidar = feats_down_lidar->points[idx + j + 1];
                    PointType &point_world = feats_down_world->points[idx + j + 1];
                    if (use_imu_as_input)
                        pointLidarToWorld(point_lidar, point_world, kf_input.x_);
                    else
                        pointLidarToWorld(point_lidar, point_world, kf_output.x_);
                }

                idx += time_seq[k];
            }
            state_in = kf_input.get_x();
        }
        else
        {
            /**** point by point update ****/
            double pcl_beg_time = measures->lidar_beg_time;
            idx = -1;
            for (k = 0; k < time_seq.size(); k++)
            {
                PointType &point_body = feats_down_lidar->points[idx + time_seq[k]];
                auto time_current = point_body.curvature / 1000.0 + pcl_beg_time;

                if (is_first_frame)
                {
                    if (imu_en)
                    {
                        while (time_current > imu_next.timestamp)
                        {
                            imu_last = imu_next;
                            imu_next = *(imu_buffer.front());
                            imu_buffer.pop_front();
                        }

                        angvel_avr = imu_last.angular_velocity;
                        acc_avr = imu_last.linear_acceleration;
                    }
                    is_first_frame = false;
                    time_update_last = time_current;
                    time_predict_last = time_current;
                }
                if (imu_en)
                {
                    bool imu_comes = time_current > imu_next.timestamp;
                    while (imu_comes)
                    {
                        angvel_avr = imu_next.angular_velocity;
                        acc_avr = imu_next.linear_acceleration;

                        /*** covariance update ***/
                        imu_last = imu_next;
                        imu_next = *(imu_buffer.front());
                        imu_buffer.pop_front();
                        double dt = imu_last.timestamp - time_predict_last;
                        kf_output.predict(dt, Q_output, input_in, true, false);
                        time_predict_last = imu_last.timestamp; // big problem
                        imu_comes = time_current > imu_next.timestamp;
                        // if (!imu_comes)
                        {
                            double dt_cov = imu_last.timestamp - time_update_last;

                            if (dt_cov > 0.0)
                            {
                                time_update_last = imu_last.timestamp;
                                // double propag_imu_start = omp_get_wtime();
                                kf_output.predict(dt_cov, Q_output, input_in, false, true);

                                // propag_time += omp_get_wtime() - propag_imu_start;
                                // double solve_imu_start = omp_get_wtime();
                                kf_output.update_iterated_dyn_share_IMU();
                                // solve_time += omp_get_wtime() - solve_imu_start;
                            }
                        }
                    }
                }

                double dt = time_current - time_predict_last;
                // double propag_state_start = omp_get_wtime();
                if (!prop_at_freq_of_imu)
                {
                    double dt_cov = time_current - time_update_last;
                    if (dt_cov > 0.0)
                    {
                        kf_output.predict(dt_cov, Q_output, input_in, false, true);
                        time_update_last = time_current;
                    }
                }
                kf_output.predict(dt, Q_output, input_in, true, false);
                // propag_time += omp_get_wtime() - propag_state_start;
                time_predict_last = time_current;

                // double t_update_start = omp_get_wtime();
                if (!kf_output.update_iterated_pointlio())
                {
                    idx += time_seq[k];
                    continue;
                }

                if (prop_at_freq_of_imu)
                {
                    double dt_cov = time_current - time_update_last;
                    if (!imu_en && (dt_cov >= 0.01)) // (point_cov_not_prop && imu_prop_cov)
                    {
                        // double propag_cov_start = omp_get_wtime();
                        kf_output.predict(dt_cov, Q_output, input_in, false, true);
                        time_update_last = time_current;
                        // propag_time += omp_get_wtime() - propag_cov_start;
                    }
                }

                // solve_start = omp_get_wtime();
                // solve_time += omp_get_wtime() - solve_start;
                // update_time += omp_get_wtime() - t_update_start;

                for (int j = 0; j < time_seq[k]; j++)
                {
                    PointType &point_lidar = feats_down_lidar->points[idx + j + 1];
                    PointType &point_world = feats_down_world->points[idx + j + 1];
                    if (use_imu_as_input)
                        pointLidarToWorld(point_lidar, point_world, kf_input.x_);
                    else
                        pointLidarToWorld(point_lidar, point_world, kf_output.x_);
                }

                idx += time_seq[k];
            }
            state_out = kf_output.get_x();
        }

        // kf.update_iterated_dyn_share_modified(lidar_meas_cov, loger.iterate_ekf_time);
        loger.meas_update_time = loger.timer.elapsedLast();
        if (use_imu_as_input)
            loger.dump_state_to_log(loger.fout_update, state_in, measures->lidar_beg_time - loger.first_lidar_beg_time);
        else
            loger.dump_state_to_log(loger.fout_update, state_out, measures->lidar_beg_time - loger.first_lidar_beg_time);

        /*** map update ***/
        V3D pos_Lidar_world;
        if (use_imu_as_input)
            pos_Lidar_world = state_in.pos + state_in.rot.normalized() * state_in.offset_T_L_I;
        else
            pos_Lidar_world = state_out.pos + state_out.rot.normalized() * state_out.offset_T_L_I;
        lasermap_fov_segment(pos_Lidar_world);
        loger.map_remove_time = loger.timer.elapsedLast();
        map_incremental();
        loger.map_incre_time = loger.timer.elapsedLast();
        loger.kdtree_size_end = ikdtree.size();
        loger.print_fastlio_cost_time();
        loger.output_fastlio_log_to_csv(measures->lidar_beg_time);
        return true;
    }

    virtual state_ikfom get_state()
    {
        state_ikfom state_ret;
        if (use_imu_as_input)
        {
            state_in = kf_input.get_x();
            state_ret.pos = state_in.pos;
            state_ret.rot = state_in.rot;
            state_ret.offset_T_L_I = state_in.offset_T_L_I;
            state_ret.offset_R_L_I = state_in.offset_R_L_I;
        }
        else
        {
            state_out = kf_output.get_x();
            state_ret.pos = state_out.pos;
            state_ret.rot = state_out.rot;
            state_ret.offset_T_L_I = state_out.offset_T_L_I;
            state_ret.offset_R_L_I = state_out.offset_R_L_I;
        }
        return state_ret;
    }

private:
    // 计算lidar point-to-plane Jacobi和残差
    void lidar_meas_model_input(state_input &state, esekfom::pointlio_datastruct<double> &ekfom_data)
    {
        normvec->resize(time_seq[k]);
        int effect_num_k = 0;
        for (int j = 0; j < time_seq[k]; j++)
        {
            PointType &point_lidar = feats_down_lidar->points[idx + j + 1];
            PointType &point_world = feats_down_world->points[idx + j + 1];
            if (use_imu_as_input)
                pointLidarToWorld(point_lidar, point_world, kf_input.x_);
            else
                pointLidarToWorld(point_lidar, point_world, kf_output.x_);
            V3D p_body = V3D(point_lidar.x, point_lidar.y, point_lidar.z);
            {
                auto &points_near = nearest_points[idx + j + 1];

                vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
                ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis, 2.236); // 1.0); //, 3.0); // 2.236;

                if ((points_near.size() < NUM_MATCH_POINTS) || pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5) // 5)
                {
                    point_matched_surface[idx + j + 1] = false;
                }
                else
                {
                    Eigen::Vector4d abcd;
                    point_matched_surface[idx + j + 1] = false;
                    float plane_thr = 0.1;   // plane_thr: the threshold for plane criteria, the smaller, the flatter a plane
                    if (esti_plane(abcd, points_near, plane_thr))
                    {
                        float dis = abcd(0) * point_world.x + abcd(1) * point_world.y + abcd(2) * point_world.z + abcd(3);
                        float s = 1 - 0.9 * fabs(dis) / sqrt(p_body.norm());

                        if (s > 0.9)
                        {
                            point_matched_surface[idx + j + 1] = true;
                            normvec->points[j].x = abcd(0);
                            normvec->points[j].y = abcd(1);
                            normvec->points[j].z = abcd(2);
                            normvec->points[j].intensity = abcd(3);
                            effect_num_k++;
                        }
                    }
                }
            }
        }
        if (effect_num_k == 0)
        {
            ekfom_data.valid = false;
            return;
        }
        ekfom_data.M_Noise = lidar_meas_cov;
        ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_num_k, 12);
        ekfom_data.z.resize(effect_num_k);
        int m = 0;
        for (int j = 0; j < time_seq[k]; j++)
        {
            if (point_matched_surface[idx + j + 1])
            {
                V3D norm_vec(normvec->points[j].x, normvec->points[j].y, normvec->points[j].z);

                if (extrinsic_est_en)
                {
                    V3D p_body = V3D(feats_down_lidar->points[idx + j + 1].x, feats_down_lidar->points[idx + j + 1].y, feats_down_lidar->points[idx + j + 1].z);
                    M3D p_crossmat, p_imu_crossmat;
                    p_crossmat = SO3Math::get_skew_symmetric(p_body);
                    V3D point_imu = state.offset_R_L_I.normalized() * p_body + state.offset_T_L_I;
                    p_imu_crossmat = SO3Math::get_skew_symmetric(point_imu);
                    V3D C(state.rot.conjugate().normalized() * norm_vec);
                    V3D A(p_imu_crossmat * C);
                    V3D B(p_crossmat * state.offset_R_L_I.conjugate().normalized() * C);
                    ekfom_data.h_x.block<1, 12>(m, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
                }
                else
                {
                    M3D point_crossmat = crossmat_list[idx + j + 1];
                    V3D C(state.rot.conjugate().normalized() * norm_vec);
                    V3D A(point_crossmat * C);
                    ekfom_data.h_x.block<1, 12>(m, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
                }
                ekfom_data.z(m) = -norm_vec(0) * feats_down_world->points[idx + j + 1].x - norm_vec(1) * feats_down_world->points[idx + j + 1].y - norm_vec(2) * feats_down_world->points[idx + j + 1].z - normvec->points[j].intensity;
                m++;
            }
        }
    }

    void lidar_meas_model_output(state_output &state, esekfom::pointlio_datastruct<double> &ekfom_data)
    {
        normvec->resize(time_seq[k]);
        int effect_num_k = 0;
        for (int j = 0; j < time_seq[k]; j++)
        {
            PointType &point_lidar = feats_down_lidar->points[idx + j + 1];
            PointType &point_world = feats_down_world->points[idx + j + 1];
            if (use_imu_as_input)
                pointLidarToWorld(point_lidar, point_world, kf_input.x_);
            else
                pointLidarToWorld(point_lidar, point_world, kf_output.x_);
            V3D p_body = V3D(point_lidar.x, point_lidar.y, point_lidar.z);
            {
                auto &points_near = nearest_points[idx + j + 1];

                vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
                ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis, 2.236);

                if ((points_near.size() < NUM_MATCH_POINTS) || pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5)
                {
                    point_matched_surface[idx + j + 1] = false;
                }
                else
                {
                    Eigen::Vector4d abcd;
                    point_matched_surface[idx + j + 1] = false;
                    float plane_thr = 0.1;   // plane_thr: the threshold for plane criteria, the smaller, the flatter a plane
                    if (esti_plane(abcd, points_near, plane_thr))
                    {
                        float dis = abcd(0) * point_world.x + abcd(1) * point_world.y + abcd(2) * point_world.z + abcd(3);
                        float s = 1 - 0.9 * fabs(dis) / sqrt(p_body.norm());

                        if (s > 0.9)
                        {
                            point_matched_surface[idx + j + 1] = true;
                            normvec->points[j].x = abcd(0);
                            normvec->points[j].y = abcd(1);
                            normvec->points[j].z = abcd(2);
                            normvec->points[j].intensity = abcd(3);
                            effect_num_k++;
                        }
                    }
                }
            }
        }
        if (effect_num_k == 0)
        {
            ekfom_data.valid = false;
            return;
        }
        ekfom_data.M_Noise = lidar_meas_cov;
        ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_num_k, 12);
        ekfom_data.z.resize(effect_num_k);
        int m = 0;
        for (int j = 0; j < time_seq[k]; j++)
        {
            if (point_matched_surface[idx + j + 1])
            {
                V3D norm_vec(normvec->points[j].x, normvec->points[j].y, normvec->points[j].z);

                if (extrinsic_est_en)
                {
                    V3D p_body = V3D(feats_down_lidar->points[idx + j + 1].x, feats_down_lidar->points[idx + j + 1].y, feats_down_lidar->points[idx + j + 1].z);
                    M3D p_crossmat, p_imu_crossmat;
                    p_crossmat = SO3Math::get_skew_symmetric(p_body);
                    V3D point_imu = state.offset_R_L_I.normalized() * p_body + state.offset_T_L_I;
                    p_imu_crossmat = SO3Math::get_skew_symmetric(point_imu);
                    V3D C(state.rot.conjugate().normalized() * norm_vec);
                    V3D A(p_imu_crossmat * C);
                    V3D B(p_crossmat * state.offset_R_L_I.conjugate().normalized() * C);
                    ekfom_data.h_x.block<1, 12>(m, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
                }
                else
                {
                    M3D point_crossmat = crossmat_list[idx + j + 1];
                    V3D C(state.rot.conjugate().normalized() * norm_vec);
                    V3D A(point_crossmat * C);
                    // V3D A(point_crossmat * state.rot_end.transpose() * norm_vec);
                    ekfom_data.h_x.block<1, 12>(m, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
                }
                ekfom_data.z(m) = -norm_vec(0) * feats_down_world->points[idx + j + 1].x - norm_vec(1) * feats_down_world->points[idx + j + 1].y - norm_vec(2) * feats_down_world->points[idx + j + 1].z - normvec->points[j].intensity;
                m++;
            }
        }
    }

    void imu_meas_model_output(state_output &state, esekfom::pointlio_datastruct<double> &ekfom_data)
    {
        std::memset(ekfom_data.satu_check, false, 6);
        ekfom_data.z_IMU.block<3, 1>(0, 0) = angvel_avr - state.omg - state.bg;
        ekfom_data.z_IMU.block<3, 1>(3, 0) = acc_avr * G_m_s2 / acc_norm - state.acc - state.ba;
        ekfom_data.R_IMU = R_imu;

        if (check_saturation)
        {
            for (auto i = 0; i < 3; ++i)
            {
                if (fabs(angvel_avr(i)) >= 0.99 * saturation_gyro)
                {
                    ekfom_data.satu_check[i] = true;
                    ekfom_data.z_IMU(i) = 0.0;
                }
            }

            for (auto i = 0; i < 3; ++i)
            {
                if (fabs(acc_avr(i)) >= 0.99 * saturation_acc)
                {
                    ekfom_data.satu_check[i + 3] = true;
                    ekfom_data.z_IMU(i + 3) = 0.0;
                }
            }
        }
    }

protected:
    virtual void map_incremental()
    {
        PointVector PointToAdd;
        PointVector PointNoNeedDownsample;
        PointToAdd.reserve(feats_down_size);
        PointNoNeedDownsample.reserve(feats_down_size);

        for (int i = 0; i < feats_down_size; i++)
        {
            if (!nearest_points[i].empty())
            {
                const PointVector &points_near = nearest_points[i];
                PointType mid_point;
                mid_point.x = floor(feats_down_world->points[i].x / ikdtree_resolution + 0.5) * ikdtree_resolution;
                mid_point.y = floor(feats_down_world->points[i].y / ikdtree_resolution + 0.5) * ikdtree_resolution;
                mid_point.z = floor(feats_down_world->points[i].z / ikdtree_resolution + 0.5) * ikdtree_resolution;
                /* If the nearest points is definitely outside the downsample box */
                if (fabs(points_near[0].x - mid_point.x) > 1.732 * ikdtree_resolution ||
                    fabs(points_near[0].y - mid_point.y) > 1.732 * ikdtree_resolution ||
                    fabs(points_near[0].z - mid_point.z) > 1.732 * ikdtree_resolution)
                {
                    PointNoNeedDownsample.emplace_back(feats_down_world->points[i]);
                    continue;
                }
                /* Check if there is a point already in the downsample box */
                bool need_add = true;
                for (int readd_i = 0; readd_i < points_near.size(); readd_i++)
                {
                    /* Those points which are outside the downsample box should not be considered. */
                    if (fabs(points_near[readd_i].x - mid_point.x) < 0.5 * ikdtree_resolution &&
                        fabs(points_near[readd_i].y - mid_point.y) < 0.5 * ikdtree_resolution &&
                        fabs(points_near[readd_i].z - mid_point.z) < 0.5 * ikdtree_resolution)
                    {
                        need_add = false;
                        break;
                    }
                }
                if (need_add)
                    PointToAdd.emplace_back(feats_down_world->points[i]);
            }
            else
            {
                // PointToAdd.emplace_back(feats_down_world->points[i]);
                PointNoNeedDownsample.emplace_back(feats_down_world->points[i]);
            }
        }

        double st_time = omp_get_wtime();
        ikdtree.Add_Points(PointToAdd, true);
        ikdtree.Add_Points(PointNoNeedDownsample, false);
        loger.add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
        loger.kdtree_incremental_time = (omp_get_wtime() - st_time) * 1000;
    }

    // 压缩相同时间的点，一起更新
    std::vector<int> time_compressing(const PointCloudType::Ptr &point_cloud)
    {
        int points_size = point_cloud->points.size();
        int j = 0;
        std::vector<int> time_seq;
        time_seq.reserve(points_size);
        for (int i = 0; i < points_size - 1; i++)
        {
            j++;
            if (point_cloud->points[i + 1].curvature > point_cloud->points[i].curvature)
            {
                time_seq.emplace_back(j);
                j = 0;
            }
        }
        if (j == 0)
        {
            time_seq.emplace_back(1);
        }
        else
        {
            time_seq.emplace_back(j + 1);
        }
        return time_seq;
    }

public:
    bool imu_en = true;
    bool use_imu_as_input = true;
    bool prop_at_freq_of_imu = true;
    bool check_saturation = true;
    double acc_norm = 9.81;

    ImuData imu_last;
    ImuData imu_next;

    bool is_first_frame = true;
    double time_update_last = 0;
    double time_predict_last = 0;
    int k;
    int idx;
    std::vector<int> time_seq;
    std::vector<M3D> crossmat_list;

    V3D angvel_avr;
    V3D acc_avr;
    double saturation_acc = 30;
    double saturation_gyro = 35;
    Eigen::Matrix<double, 6, 1> R_imu;

    Eigen::Matrix<double, 24, 24> Q_input;
    Eigen::Matrix<double, 30, 30> Q_output;

private:
    esekfom::esekf<state_input, 24, input_ikfom> kf_input;
    esekfom::esekf<state_output, 30, input_ikfom> kf_output;
    state_input state_in;
    state_output state_out;
    input_ikfom input_in;
};
