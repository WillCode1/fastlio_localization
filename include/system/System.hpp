#pragma once
#include <omp.h>
#include <math.h>
#include <thread>
#include <pcl/io/pcd_io.h>
#include "ikd-Tree/ikd_Tree.h"
#include "frontend/FastlioOdometry.hpp"
#include "frontend/PointlioOdometry.hpp"
#include "utility/Header.h"
#include "Relocalization.hpp"

class System
{
public:
    System()
    {
        relocalization = make_shared<Relocalization>();

        feats_undistort.reset(new PointCloudType());
    }

    void init_system_mode()
    {
        frontend->detect_range = frontend->lidar->detect_range;
        frontend->init_estimator();

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

    bool run()
    {
        if (!frontend->run(feats_undistort))
        {
            system_state_vaild = false;
            return system_state_vaild;
        }
        else if (feats_undistort->empty() || (feats_undistort == NULL))
        {
            return false;
        }

        if (check_state_nan(frontend->get_state()))
        {
            system_state_vaild = false;
            return system_state_vaild;
        }

        system_state_vaild = true;
        return system_state_vaild;
    }

    bool run_relocalization(PointCloudType::Ptr scan)
    {
        run_relocalization_thread = true;
        if (!system_state_vaild)
        {
            Eigen::Matrix4d imu_pose;
            if (relocalization->run(scan, imu_pose))
            {
                frontend->reset_state(imu_pose);
                system_state_vaild = true;
            }
            else
            {
#ifdef DEDUB_MODE
                frontend->reset_state(imu_pose);
#endif
            }
        }
        run_relocalization_thread = false;
        return system_state_vaild;
    }

public:
    bool system_state_vaild = false; // true: system ok
    bool run_relocalization_thread = false;
    std::thread relocalization_thread;

    /*** module ***/
    shared_ptr<FastlioOdometry> frontend;
    shared_ptr<Relocalization> relocalization;

    /*** keyframe config ***/
    PointCloudType::Ptr feats_undistort;

    /*** global map maintain ***/
    string globalmap_path = PCD_FILE_DIR("globalmap.pcd");
    string trajectory_path = PCD_FILE_DIR("trajectory.pcd");
    string scd_path = PCD_FILE_DIR("scancontext/");
};
