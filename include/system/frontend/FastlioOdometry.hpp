#pragma once
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include "ImuProcessor.h"
#include "LidarProcessor.hpp"
#include "system/Header.h"

#define USE_IKDTREE

class FastlioOdometry
{
public:
    FastlioOdometry() : frontend_type(Fastlio)
    {
        surf_frame_ds_res = 0.5;
        point_skip_num = 10;
        ikdtree_resolution = 0.5;
        cube_len = 200;
        detect_range = 300;

        feats_down_lidar.reset(new PointCloudType());
        feats_down_world.reset(new PointCloudType());
        normvec.reset(new PointCloudType());

        lidar = make_shared<LidarProcessor>();
        imu = make_shared<ImuProcessor>();
        measures = make_shared<MeasureCollection>();
    }

    virtual ~FastlioOdometry() {}

    virtual void init_estimator()
    {
        double epsi[23] = {0.001};
        fill(epsi, epsi + 23, 0.001);
        auto lidar_meas_model = [&](state_ikfom &a, esekfom::fastlio_datastruct<double> &b) { this->lidar_meas_model(a, b); };
        kf.init_dyn_share(get_f, df_dx, df_dw, lidar_meas_model, num_max_iterations, epsi);

        Eigen::Matrix<double, 23, 23> init_P;
        init_P.setIdentity();
        init_P.block<3, 3>(6, 6).diagonal() << 0.00001, 0.00001, 0.00001; // lidar和imu外参旋转量协方差
        init_P.block<3, 3>(9, 9).diagonal() << 0.00001, 0.00001, 0.00001; // lidar和imu外参平移量协方差
        init_P.block<3, 3>(15, 15).diagonal() << 0.0001, 0.0001, 0.0001;  // 陀螺仪偏差协方差
        init_P.block<3, 3>(18, 18).diagonal() << 0.001, 0.001, 0.001;     // 加速度偏差协方差
        init_P.block<2, 2>(21, 21).diagonal() << 0.00001, 0.00001;        // 重力协方差
        kf.change_P(init_P);
    }

    virtual void set_extrinsic(const V3D &transl, const M3D &rot)
    {
        // imu = R * lidar + t
        state = kf.get_x();
        state.offset_T_L_I = transl;
        state.offset_R_L_I = rot;
        kf.change_x(state);

        offset_Tli = transl;
        offset_Rli = rot;
    }

    virtual void init_state(shared_ptr<ImuProcessor> &imu)
    {
        state.grav.vec << VEC_FROM_ARRAY(gravity_init);
        state.bg = imu->mean_gyr;
        kf.change_x(state);
    }

    virtual void reset_state(const Eigen::Matrix4d &imu_pose)
    {
        Eigen::Quaterniond fine_tune_quat(M3D(imu_pose.topLeftCorner(3, 3)));
        state = kf.get_x();
        state.vel.setZero();
        state.ba.setZero();
        state.bg.setZero();
        state.offset_R_L_I = offset_Rli;
        state.offset_T_L_I = offset_Tli;
        state.grav.vec << VEC_FROM_ARRAY(gravity_init);
        state.pos = V3D(imu_pose.topRightCorner(3, 1));
        state.rot.coeffs() = Vector4d(fine_tune_quat.x(), fine_tune_quat.y(), fine_tune_quat.z(), fine_tune_quat.w());
        imu_state = state;
        kf.change_x(state);
    }

    virtual void cache_imu_data(double timestamp, const V3D &angular_velocity, const V3D &linear_acceleration)
    {
        timestamp = timestamp + timedelay_lidar2imu; // 时钟同步该采样同步td
        std::lock_guard<std::mutex> lock(mtx_buffer);

        if (timestamp < latest_timestamp_imu)
        {
            LOG_ERROR("imu loop back, clear buffer");
            return;
        }

        latest_timestamp_imu = timestamp;
        imu_buffer.push_back(make_shared<ImuData>(latest_timestamp_imu, angular_velocity, linear_acceleration));
    }

    virtual void cache_pointcloud_data(const double &lidar_beg_time, const PointCloudType::Ptr &scan)
    {
        std::lock_guard<std::mutex> lock(mtx_buffer);
        if (lidar_beg_time < latest_lidar_beg_time)
        {
            LOG_ERROR("lidar loop back, clear buffer");
            return;
        }

        if (scan->points.size() <= 1)
        {
            LOG_WARN("Too few input point cloud! size = %ld, scan droped!", scan->points.size());
            return;
        }

        latest_lidar_beg_time = lidar_beg_time;
        double latest_lidar_end_time = latest_lidar_beg_time + scan->points.back().curvature / 1000;

        if (abs(latest_lidar_end_time - latest_timestamp_imu) > 1)
        {
            LOG_WARN("IMU and LiDAR's clock not synced, IMU time: %lf, lidar time: %lf. Maybe set timedelay_lidar2imu = %lf.\n",
                     latest_timestamp_imu, latest_lidar_end_time, latest_lidar_end_time - latest_timestamp_imu);
        }

        lidar_buffer.push_back(scan);
        time_buffer.push_back(latest_lidar_beg_time);
    }

    virtual bool sync_sensor_data()
    {
        static bool lidar_pushed = false;

        std::lock_guard<std::mutex> lock(mtx_buffer);
        if (lidar_buffer.empty() || imu_buffer.empty())
        {
            return false;
        }

        /*** push a lidar scan ***/
        if (!lidar_pushed)
        {
            measures->lidar = lidar_buffer.front();
            measures->lidar_beg_time = time_buffer.front();
            sort(measures->lidar->points.begin(), measures->lidar->points.end(), compare_timestamp);
            lidar_end_time = measures->lidar_beg_time + measures->lidar->points.back().curvature / double(1000);
            measures->lidar_end_time = lidar_end_time;
            lidar_pushed = true;
        }

        if (latest_timestamp_imu < lidar_end_time)
        {
            return false;
        }

        /*** push imu data, and pop from imu buffer ***/
        measures->imu.clear();
        while (!imu_buffer.empty())
        {
            if (imu_buffer.front()->timestamp > lidar_end_time)
                break;
            measures->imu.push_back(imu_buffer.front());
#define RECORD_IMU_STATE
#ifdef RECORD_IMU_STATE
            angular_velocity = imu_buffer.front()->angular_velocity;
            linear_acceleration = imu_buffer.front()->linear_acceleration;
#endif
            imu_buffer.pop_front();
        }

        lidar_buffer.pop_front();
        time_buffer.pop_front();
        lidar_pushed = false;
        return true;
    }

    virtual bool run(PointCloudType::Ptr &feats_undistort)
    {
#ifndef NO_LOGER
        if (loger.runtime_log && !loger.inited_first_lidar_beg_time)
        {
            loger.first_lidar_beg_time = measures->lidar_beg_time;
            loger.inited_first_lidar_beg_time = true;
        }
#endif
        LOG_DEBUG("run fastlio 1");
        loger.resetTimer();
        feats_down_lidar->clear();
        for (int i = 0; i < measures->lidar->points.size(); i++)
            if (i % point_skip_num == 0)
            {
                feats_down_lidar->points.push_back(measures->lidar->points[i]);
            }
        measures->lidar = feats_down_lidar;
        LOG_DEBUG("run fastlio 2");
        imu->Process(*measures, kf, feats_undistort);
        LOG_DEBUG("run fastlio 3");

        if (feats_undistort->empty() || (feats_undistort == NULL))
        {
            LOG_WARN("Wait for the imu to initialize completed!");
            return true;
        }

        // if (!imu->gravity_align_)
            // init_state(imu);

        state = kf.get_x();
        loger.imu_process_time = loger.timer.elapsedLast();
        loger.feats_undistort_size = feats_undistort->points.size();

        LOG_DEBUG("run fastlio 4");
        /*** interval sample and downsample the feature points in a scan ***/
        if (space_down_sample)
        {
            surf_frame_ds_filter.setLeafSize(surf_frame_ds_res, surf_frame_ds_res, surf_frame_ds_res);
            surf_frame_ds_filter.setInputCloud(feats_undistort);
            surf_frame_ds_filter.filter(*feats_down_lidar);
        }
        feats_down_size = feats_down_lidar->points.size();
        loger.feats_down_size = feats_down_size;
        loger.downsample_time = loger.timer.elapsedLast();

        // loger.kdtree_size = ikdtree.size();

        if (feats_down_size < 5)
        {
            LOG_WARN("No point, skip this scan!");
            return true;
        }

#ifndef NO_LOGER
        loger.dump_state_to_log(loger.fout_predict, state, measures->lidar_beg_time - loger.first_lidar_beg_time);
#endif

        LOG_DEBUG("run fastlio 5");
        /*** iterated state estimation ***/
        feats_down_world->resize(feats_down_size);
        point_matched_surface.resize(feats_down_size);
        nearest_points.resize(feats_down_size);
        normvec->resize(feats_down_size);
        bool measure_valid = true, iter_converge = false;
        kf.update_iterated_fastlio2(measure_valid, iter_converge);
        if (!measure_valid)
        {
            LOG_ERROR("Lidar degradation!");
            return false;
        }
        else if (!iter_converge && check_for_not_converged(measures->lidar_beg_time, not_converged_times_thold))
        {
            LOG_ERROR("Iteration doesn't converge beyond the limit, reset the system!");
            return false;
        }
        state = kf.get_x();
        imu_state = state;
        loger.meas_update_time = loger.timer.elapsedLast();
#ifndef NO_LOGER
        loger.dump_state_to_log(loger.fout_update, state, measures->lidar_beg_time - loger.first_lidar_beg_time);
#endif

#ifdef USE_IKDTREE
        if (false)
        {
            /*** map update ***/
            // TODO: location mode dongtai load submap, and when not in raw map, could add new scan into ikdtree.
            V3D pos_Lidar_world = state.pos + state.rot * state.offset_T_L_I;
            lasermap_fov_segment(pos_Lidar_world);
            loger.map_remove_time = loger.timer.elapsedLast();
            map_incremental();
            loger.map_incre_time = loger.timer.elapsedLast();
            // loger.kdtree_size_end = ikdtree.size();
        }
#endif
        loger.print_fastlio_cost_time();
#ifndef NO_LOGER
        loger.output_fastlio_log_to_csv(measures->lidar_beg_time);
#endif
        LOG_DEBUG("run fastlio 6");
        return true;
    }

public:
    void init_global_map(PointCloudType::Ptr &submap)
    {
#ifdef USE_IKDTREE
        if (ikdtree.Root_Node != nullptr)
        {
            LOG_ERROR("Error, ikdtree not null when initializing the map!");
            std::exit(100);
        }
        ikdtree.set_downsample_param(ikdtree_resolution);
        ikdtree.Build(submap->points);
#else
        if (kdtree.getInputCloud() != nullptr)
        {
            LOG_ERROR("Error, kdtree not null when initializing the map!");
            std::exit(100);
        }
        kdtree.setInputCloud(submap);
#endif
    }

    virtual state_ikfom get_state()
    {
        state = kf.get_x();
        return state;
    }

    virtual state_ikfom integrate_imu_odom(const ImuData &imu_meas)
    {
        static double last_time = 0;
        V3D last_acc;
        V3D last_gyro;

        double dt = 0;
        input_ikfom in;
        if (last_time == 0)
        {
            in.acc = imu_meas.linear_acceleration;
            in.gyro = imu_meas.angular_velocity;
            dt = 1.0 / imu->imu_rate;
        }
        else
        {
            in.acc = 0.5 * (imu_meas.linear_acceleration + last_acc);
            in.gyro = 0.5 * (imu_meas.angular_velocity + last_gyro);
            dt = imu_meas.timestamp - last_time;
        }
        Eigen::Matrix<double, 24, 1> f = get_f(imu_state, in);

        imu_state.oplus(f, dt);
        last_time = imu_meas.timestamp;
        last_acc = imu_meas.linear_acceleration;
        last_gyro = imu_meas.angular_velocity;
        return imu_state;
    }

private:
    // 计算lidar point-to-plane Jacobi和残差
    void lidar_meas_model(state_ikfom &state, esekfom::fastlio_datastruct<double> &ekfom_data)
    {
        normvec->clear();
        effect_features.clear();

        QD lidar_rot = state.rot * state.offset_R_L_I;
        V3D lidar_pos = state.rot * state.offset_T_L_I + state.pos;
        /** closest surface search and residual computation **/
#ifdef MP_EN
#pragma omp parallel for num_threads(MP_PROC_NUM)
#endif
        for (int i = 0; i < feats_down_size; i++)
        {
            /* transform to world frame */
            PointType point_world;
            pointLidarToWorld(feats_down_lidar->points[i], point_world, lidar_rot, lidar_pos);

            auto &points_near = nearest_points[i];

            if (ekfom_data.converge)
            {
                /** Find the closest surfaces in the map **/
                vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
#ifdef USE_IKDTREE
                ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis, lidar_model_search_range);
                point_matched_surface[i] = points_near.size() < NUM_MATCH_POINTS ? false : true;
#else
                vector<int> indices;
                kdtree.nearestKSearch(point_world, NUM_MATCH_POINTS, indices, pointSearchSqDis);
                point_matched_surface[i] = indices.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis.back() > lidar_model_search_range ? false : true;
                if (point_matched_surface[i])
                {
                    points_near.clear();
                    for (auto index: indices)
                        points_near.push_back(kdtree.getInputCloud()->points[index]);
                }
#endif
            }

            if (!point_matched_surface[i])
                continue;

            Eigen::Vector4d abcd;
            point_matched_surface[i] = false;
            if (esti_plane(abcd, points_near, 0.1f))
            {
                // abcd 分别为 Ax + By + Cz + D = 0 中的系数，dis = |Ax + By + Cz + D| / sqrt(A^2 + B^2 + C^2)
                // A、B、C 是平面的法向量，D 是原点离平面的有符号距离.
                float dis = abcd(0) * point_world.x + abcd(1) * point_world.y + abcd(2) * point_world.z + abcd(3);
                // 1.点到面距离越大，可能性越小
                // 2.但是根据雷达物理模型原理，远处点的可以放宽一些
                float s = 1 - 0.9 * fabs(dis) / sqrt(pointDistance(feats_down_lidar->points[i]));

                if (s > 0.9)
                {
                    point_matched_surface[i] = true;
                    normvec->points[i].x = abcd(0);
                    normvec->points[i].y = abcd(1);
                    normvec->points[i].z = abcd(2);
                    normvec->points[i].intensity = dis;
                }
            }
        }

        // omp中无法push_back
        for (int i = 0; i < feats_down_size; i++)
        {
            if (point_matched_surface[i])
            {
                EffectFeature effect_feat;
                effect_feat.point_lidar = V3D(feats_down_lidar->points[i].x, feats_down_lidar->points[i].y, feats_down_lidar->points[i].z);
                effect_feat.norm_vec = V3D(normvec->points[i].x, normvec->points[i].y, normvec->points[i].z);
                effect_feat.residual = normvec->points[i].intensity;
                effect_features.emplace_back(effect_feat);
            }
        }

        int effct_feat_num = effect_features.size();
        if (effct_feat_num < 1)
        {
            ekfom_data.valid = false;
            LOG_WARN("No Effective Points!");
            return;
        }

        /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
        ekfom_data.H = MatrixXd::Zero(effct_feat_num, 12);
        ekfom_data.z.resize(effct_feat_num);

        // 求观测值与误差的雅克比矩阵，如论文式14以及式12、13
#ifdef MP_EN
#pragma omp parallel for num_threads(MP_PROC_NUM)
#endif
        for (int i = 0; i < effct_feat_num; i++)
        {
            const V3D &point_lidar = effect_features[i].point_lidar;
            const M3D &point_be_crossmat = hat(point_lidar);
            const V3D &point_imu = state.offset_R_L_I * point_lidar + state.offset_T_L_I;
            const M3D &point_crossmat = hat(point_imu);

            /*** get the normal vector of closest surface ***/
            const V3D &norm_vec = effect_features[i].norm_vec;

            /*** calculate the Measuremnt Jacobian matrix H ***/
            // 雅各比矩阵分子布局和分母布局的区别：(AR^Tu)^T = u^TR(A^T) = u^TR(-A)
            V3D C(state.rot.conjugate() * norm_vec);
            V3D A(point_crossmat * C);
            if (extrinsic_est_en)
            {
                V3D B(point_be_crossmat * state.offset_R_L_I.conjugate() * C);
                ekfom_data.H.block<1, 12>(i, 0) << VEC_FROM_ARRAY(norm_vec), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
            }
            else
            {
                ekfom_data.H.block<1, 12>(i, 0) << VEC_FROM_ARRAY(norm_vec), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            }

            /*** Measuremnt: distance to the closest plane ***/
            ekfom_data.z(i) = -effect_features[i].residual;
        }
        ekfom_data.R = lidar_meas_cov;
    }

protected:
    /**
     * 动态调整地图:
     * 1.初始化局部地图包围盒角点(首次)
     * 2.判断是否需要move
     * 3.需要就更新包围盒，并用ikdtree删除不需要的包围盒
     */
    virtual void lasermap_fov_segment(const V3D &pos_Lidar_world)
    {
        vector<BoxPointType> cub_needrm;

        // 初始化局部地图包围盒角点，以为w系下lidar位置为中心,得到长宽高200*200*200的局部地图
        if (!localmap_initialized)
        {
            for (int i = 0; i < 3; i++)
            {
                local_map_bbox.vertex_min[i] = pos_Lidar_world(i) - cube_len / 2.0;
                local_map_bbox.vertex_max[i] = pos_Lidar_world(i) + cube_len / 2.0;
            }
            localmap_initialized = true;
            return;
        }
        // 各个方向上Lidar与局部地图边界的距离，或者说是lidar与立方体盒子六个面的距离
        float dist_to_map_edge[3][2];
        bool need_move = false;
        // 当前雷达系中心到各个地图边缘的距离
        for (int i = 0; i < 3; i++)
        {
            dist_to_map_edge[i][0] = fabs(pos_Lidar_world(i) - local_map_bbox.vertex_min[i]);
            dist_to_map_edge[i][1] = fabs(pos_Lidar_world(i) - local_map_bbox.vertex_max[i]);
            // 与某个方向上的边界距离太小，标记需要移除need_move，参考论文Fig3
            if (dist_to_map_edge[i][0] <= move_threshold * detect_range || dist_to_map_edge[i][1] <= move_threshold * detect_range)
                need_move = true;
        }
        if (!need_move)
            return;
        // 否则需要计算移动的距离
        BoxPointType New_LocalMap_Points, tmp_boxpoints;
        // 新的局部地图盒子边界点
        New_LocalMap_Points = local_map_bbox;
        float mov_dist = max((cube_len - 2.0 * move_threshold * detect_range) * 0.5 * 0.9, double(detect_range * (move_threshold - 1)));
        for (int i = 0; i < 3; i++)
        {
            tmp_boxpoints = local_map_bbox;
            if (dist_to_map_edge[i][0] <= move_threshold * detect_range)
            {
                New_LocalMap_Points.vertex_max[i] -= mov_dist;
                New_LocalMap_Points.vertex_min[i] -= mov_dist;
                tmp_boxpoints.vertex_min[i] = local_map_bbox.vertex_max[i] - mov_dist;
                cub_needrm.push_back(tmp_boxpoints); // 移除较远包围盒
            }
            else if (dist_to_map_edge[i][1] <= move_threshold * detect_range)
            {
                New_LocalMap_Points.vertex_max[i] += mov_dist;
                New_LocalMap_Points.vertex_min[i] += mov_dist;
                tmp_boxpoints.vertex_max[i] = local_map_bbox.vertex_min[i] + mov_dist;
                cub_needrm.push_back(tmp_boxpoints);
            }
        }
        local_map_bbox = New_LocalMap_Points;

        if (cub_needrm.size() > 0)
            ikdtree.Delete_Point_Boxes(cub_needrm);
    }

    virtual void map_incremental()
    {
        PointVector PointToAdd;
        PointVector PointNoNeedDownsample;
        PointToAdd.reserve(feats_down_size);
        PointNoNeedDownsample.reserve(feats_down_size);

        QD lidar_rot = state.rot * state.offset_R_L_I;
        V3D lidar_pos = state.rot * state.offset_T_L_I + state.pos;
        for (int i = 0; i < feats_down_size; i++)
        {
            pointLidarToWorld(feats_down_lidar->points[i], feats_down_world->points[i], lidar_rot, lidar_pos);
            if (!nearest_points[i].empty())
            {
                const PointVector &points_near = nearest_points[i];
                PointType mid_point;
                mid_point.x = (floor(feats_down_world->points[i].x / ikdtree_resolution) + 0.5) * ikdtree_resolution;
                mid_point.y = (floor(feats_down_world->points[i].y / ikdtree_resolution) + 0.5) * ikdtree_resolution;
                mid_point.z = (floor(feats_down_world->points[i].z / ikdtree_resolution) + 0.5) * ikdtree_resolution;

                if (fabs(points_near[0].x - mid_point.x) > 0.5 * ikdtree_resolution &&
                    fabs(points_near[0].y - mid_point.y) > 0.5 * ikdtree_resolution &&
                    fabs(points_near[0].z - mid_point.z) > 0.5 * ikdtree_resolution)
                {
                    PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                    continue;
                }

                bool need_add = true;
                float dist = pointDistanceSquare(feats_down_world->points[i], mid_point);
                for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++)
                {
                    if (points_near.size() < NUM_MATCH_POINTS)
                        break;
                    if (pointDistanceSquare(points_near[readd_i], mid_point) < dist)
                    {
                        need_add = false;
                        break;
                    }
                }
                if (need_add)
                    PointToAdd.push_back(feats_down_world->points[i]);
            }
            else
            {
                PointToAdd.push_back(feats_down_world->points[i]);
            }
        }

        ikdtree.Add_Points(PointToAdd, true);
        ikdtree.Add_Points(PointNoNeedDownsample, false);
    }

    bool esti_plane(Vector4d &pca_result, const PointVector &point, const double &threshold)
    {
        int num_match_point = point.size();
        MatrixXd A(num_match_point, 3);
        MatrixXd b(num_match_point, 1);
        A.setZero();
        b.setOnes();
        b *= -1.0f;

        for (int j = 0; j < num_match_point; j++)
        {
            A(j, 0) = point[j].x;
            A(j, 1) = point[j].y;
            A(j, 2) = point[j].z;
        }

        V3D normvec = A.colPivHouseholderQr().solve(b);

        auto n = normvec.norm();
        pca_result(0) = normvec(0) / n;
        pca_result(1) = normvec(1) / n;
        pca_result(2) = normvec(2) / n;
        pca_result(3) = 1.0 / n;

        for (int j = 0; j < num_match_point; j++)
        {
            if (fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y + pca_result(2) * point[j].z + pca_result(3)) > threshold)
            {
                return false;
            }
        }
        return true;
    }

    struct EffectFeature
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        V3D point_lidar;
        V3D norm_vec;
        double residual;
    };

    vector<bool> point_matched_surface;
    vector<PointVector> nearest_points;
    PointCloudType::Ptr normvec;
    std::vector<EffectFeature> effect_features;

public:
    bool extrinsic_est_en = false;
    /*** for gravity align ***/
    std::vector<double> gravity_init;

    /*** backup for relocalization reset ***/
    V3D offset_Tli;
    M3D offset_Rli;
    LogAnalysis loger;

    /*** sensor data processor ***/
    shared_ptr<LidarProcessor> lidar;
    shared_ptr<ImuProcessor> imu;
    shared_ptr<MeasureCollection> measures;
    deque<ImuData::Ptr> imu_buffer;
    deque<double> time_buffer;
    deque<PointCloudType::Ptr> lidar_buffer;

    double latest_lidar_beg_time = 0;
    double latest_timestamp_imu = -1.0;
    double timedelay_lidar2imu = 0.0;
    double lidar_end_time = 0;
    mutex mtx_buffer;

    int frontend_type;
    /*** frontend odometry ***/
    int point_skip_num;
    bool space_down_sample = true;
    double surf_frame_ds_res;
    pcl::VoxelGrid<PointType> surf_frame_ds_filter;
    int feats_down_size = 0;
    PointCloudType::Ptr feats_down_lidar;
    PointCloudType::Ptr feats_down_world;
    const int not_converged_times_thold = 5;

    /*** ESKF inputs and output ***/
    int num_max_iterations = 4;
    const int NUM_MATCH_POINTS = 5;
    double lidar_model_search_range = 5;
    double lidar_meas_cov = 0.001;

    /*** local map maintain ***/
    bool localmap_initialized = false;
    const float move_threshold = 1.5f;
    double cube_len;
    double detect_range;
    BoxPointType local_map_bbox;
    double ikdtree_resolution;
    KD_TREE<PointType> ikdtree;
    pcl::KdTreeFLANN<PointType> kdtree;

#ifdef RECORD_IMU_STATE
    V3D angular_velocity;
    V3D linear_acceleration;
#endif

private:
    esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
    state_ikfom state;

    /*** for imu odom ***/
    state_ikfom imu_state;
};
