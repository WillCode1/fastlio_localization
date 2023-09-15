#pragma once
#include <omp.h>
#include <math.h>
#include <thread>
#include <mutex>
#include <thread>
#include <pcl/io/pcd_io.h>
#include "ikd-Tree/ikd_Tree.h"
#include "ImuProcessor.h"
#include "LidarProcessor.hpp"
#include "FastlioOdometry.hpp"
#include "Relocalization.hpp"
#include "utility/Header.h"

class System
{
public:
    System()
    {
        lidar = make_shared<LidarProcessor>();
        imu = make_shared<ImuProcessor>();

        measures = make_shared<MeasureCollection>();
        frontend = make_shared<FastlioOdometry>();
        relocalization = make_shared<Relocalization>();

        feats_undistort.reset(new PointCloudType());

        file_pose_unoptimized = fopen(DEBUG_FILE_DIR("keyframe_pose.txt").c_str(), "w");
        fprintf(file_pose_unoptimized, "# keyframe trajectory unoptimized\n# timestamp tx ty tz qx qy qz qw\n");
    }

    ~System()
    {
        fclose(file_pose_unoptimized);
    }

    void init_system_mode()
    {
        frontend->detect_range = lidar->detect_range;

        double epsi[23] = {0.001};
        fill(epsi, epsi + 23, 0.001);
        auto lidar_meas_model = [&](state_ikfom &a, esekfom::dyn_share_datastruct<double> &b) { frontend->lidar_meas_model(a, b, loger); };
        frontend->kf.init_dyn_share(get_f, df_dx, df_dw, lidar_meas_model, frontend->num_max_iterations, epsi);

        /*** init localization mode ***/
        if (access(globalmap_path.c_str(), F_OK) != 0)
        {
            LOG_ERROR("File not exist! Please check the \"globalmap_path\".");
            std::exit(100);
        }

        PointCloudType::Ptr global_map(new PointCloudType());
        Timer timer;
        pcl::io::loadPCDFile(globalmap_path, *global_map);
        if (global_map->points.size() < 5000)
        {
            LOG_ERROR("Too few point clouds! Please check the map file.");
            std::exit(100);
        }
        LOG_WARN("Load pcd successfully! There are %lu points in map. Cost time %fms.", global_map->points.size(), timer.elapsedLast());

        if (!relocalization->load_prior_map(global_map))
        {
            std::exit(100);
        }

        pcl::io::loadPCDFile(trajectory_path, *relocalization->trajectory_poses);
        if (relocalization->trajectory_poses->points.size() < 10)
        {
            LOG_ERROR("Too few point clouds! Please check the trajectory file.");
            std::exit(100);
        }
        LOG_WARN("Load trajectory poses successfully! There are %lu poses.", relocalization->trajectory_poses->points.size());

        if (!relocalization->load_keyframe_descriptor(scd_path))
        {
            LOG_ERROR("Load keyframe descriptor failed!");
            std::exit(100);
        }
        LOG_WARN("Load keyframe descriptor successfully! There are %lu descriptors.", relocalization->sc_manager->polarcontexts_.size());

        /*** initialize the map kdtree ***/
        frontend->init_global_map(global_map);
    }

    void cache_imu_data(double timestamp, const V3D &angular_velocity, const V3D &linear_acceleration)
    {
        timestamp = timestamp + timedelay_lidar2imu; // 时钟同步该采样同步td
        std::lock_guard<std::mutex> lock(mtx_buffer);

        if (timestamp < latest_timestamp_imu)
        {
            LOG_WARN("imu loop back, clear buffer");
            imu->imu_buffer.clear();
        }

        latest_timestamp_imu = timestamp;
        imu->imu_buffer.push_back(make_shared<ImuData>(latest_timestamp_imu, angular_velocity, linear_acceleration));
    }

    void cache_pointcloud_data(const double &lidar_beg_time, const PointCloudType::Ptr &scan)
    {
        std::lock_guard<std::mutex> lock(mtx_buffer);
        if (lidar_beg_time < latest_lidar_beg_time)
        {
            LOG_ERROR("lidar loop back, clear buffer");
            lidar->lidar_buffer.clear();
        }

        latest_lidar_beg_time = lidar_beg_time;
        double latest_lidar_end_time = latest_lidar_beg_time + scan->points.back().curvature / 1000;

        if (abs(latest_lidar_end_time - latest_timestamp_imu) > 1)
        {
            LOG_WARN("IMU and LiDAR's clock not synced, IMU time: %lf, lidar time: %lf. Maybe set timedelay_lidar2imu = %lf.\n",
                     latest_timestamp_imu, latest_lidar_end_time, latest_lidar_end_time - latest_timestamp_imu);
        }

        lidar->lidar_buffer.push_back(scan);
        lidar->time_buffer.push_back(latest_lidar_beg_time);
    }

    // 同步得到，当前帧激光点的开始和结束时间里的所有imu数据
    bool sync_sensor_data()
    {
        static bool lidar_pushed = false;
        static double lidar_mean_scantime = 0.0;
        static int scan_num = 0;

        std::lock_guard<std::mutex> lock(mtx_buffer);
        if (lidar->lidar_buffer.empty() || imu->imu_buffer.empty())
        {
            return false;
        }

        /*** push a lidar scan ***/
        if (!lidar_pushed)
        {
            measures->lidar = lidar->lidar_buffer.front();
            measures->lidar_beg_time = lidar->time_buffer.front();
            if (measures->lidar->points.size() <= 1) // time too little
            {
                lidar_end_time = measures->lidar_beg_time + lidar_mean_scantime;
                LOG_WARN("Too few input point cloud!\n");
            }
            else if (measures->lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
            {
                lidar_end_time = measures->lidar_beg_time + lidar_mean_scantime;
            }
            else
            {
                scan_num++;
                lidar_end_time = measures->lidar_beg_time + measures->lidar->points.back().curvature / double(1000);
                lidar_mean_scantime += (measures->lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
            }

            measures->lidar_end_time = lidar_end_time;

            lidar_pushed = true;
        }

        if (latest_timestamp_imu < lidar_end_time)
        {
            return false;
        }

        /*** push imu data, and pop from imu buffer ***/
        double imu_time = imu->imu_buffer.front()->timestamp;
        measures->imu.clear();
        while ((!imu->imu_buffer.empty()) && (imu_time <= lidar_end_time))
        {
            measures->imu.push_back(imu->imu_buffer.front());
            imu->imu_buffer.pop_front();
            imu_time = imu->imu_buffer.front()->timestamp;
        }

        lidar->lidar_buffer.pop_front();
        lidar->time_buffer.pop_front();
        lidar_pushed = false;
        return true;
    }

    bool run()
    {
        if (loger.runtime_log && !loger.inited_first_lidar_beg_time)
        {
            loger.first_lidar_beg_time = measures->lidar_beg_time;
            loger.inited_first_lidar_beg_time = true;
        }

        /*** relocalization for localization mode ***/
        if (!system_state_vaild)
        {
            Eigen::Matrix4d imu_pose;
            if (relocalization->run(measures->lidar, imu_pose))
            {
                frontend->reset_state(imu_pose);
                system_state_vaild = true;
            }
            else
            {
#ifdef DEDUB_MODE
                frontend->reset_state(imu_pose);
#endif
                system_state_vaild = false;
                return system_state_vaild;
            }
        }

        /*** frontend ***/
        loger.resetTimer();
        if (!frontend->run(imu, *measures, feats_undistort, loger))
        {
            system_state_vaild = false;
            return system_state_vaild;
        }
        else if (feats_undistort->empty() || (feats_undistort == NULL))
        {
            return false;
        }

        loger.print_fastlio_cost_time();
        loger.output_fastlio_log_to_csv(measures->lidar_beg_time);
#if 0
        // for test
        loger.save_trajectory(file_pose_unoptimized, frontend->state.pos, frontend->state.rot, measures->lidar_end_time);
#endif

        system_state_vaild = true;
        return system_state_vaild;
    }

public:
    bool system_state_vaild = false; // true: system ok
    LogAnalysis loger;

    /*** sensor data processor ***/
    shared_ptr<LidarProcessor> lidar;
    shared_ptr<ImuProcessor> imu;

    double latest_lidar_beg_time = 0;
    double latest_timestamp_imu = -1.0;
    double timedelay_lidar2imu = 0.0;
    double lidar_end_time = 0;
    mutex mtx_buffer;

    /*** module ***/
    shared_ptr<MeasureCollection> measures;
    shared_ptr<FastlioOdometry> frontend;
    shared_ptr<Relocalization> relocalization;

    /*** keyframe config ***/
    FILE *file_pose_unoptimized;
    PointCloudType::Ptr feats_undistort;

    /*** global map maintain ***/
    string globalmap_path = PCD_FILE_DIR("globalmap.pcd");
    string trajectory_path = PCD_FILE_DIR("trajectory.pcd");
    string scd_path = PCD_FILE_DIR("scancontext/");
};
