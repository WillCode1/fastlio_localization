#pragma once
#include <omp.h>
#include <math.h>
#include <thread>
#include "ikd-Tree/ikd_Tree.h"
#include "frontend/FastlioOdometry.hpp"
#include "frontend/PointlioOdometry.hpp"
#include "system/Header.h"
#include "Relocalization.hpp"
#include "utility/Pcd2Pgm.hpp"

class System
{
public:
    System()
    {
        relocalization = make_shared<Relocalization>();

        feats_undistort.reset(new PointCloudType());
        global_map.reset(new PointCloudType());
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

        if (!relocalization->bbs3d.load_config())
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
            relocalization->algorithm_type = "manually_set";
            LOG_ERROR("Load keyframe descriptor failed, set algorithm_type to manually_set!");
        }
        else
            LOG_WARN("Load keyframe descriptor successfully! There are %lu descriptors.", relocalization->sc_manager->polarcontexts_.size());

#if 0
        p2p = make_shared<Pcd2Pgm>(0.05, map_path + "/map");
        p2p->convert_from_pgm();
#endif

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

    bool run_relocalization(PointCloudType::Ptr scan, const double &lidar_beg_time)
    {
        run_relocalization_thread = true;
        if (!system_state_vaild)
        {
            Eigen::Matrix4d imu_pose;
            if (relocalization->run(scan, imu_pose, lidar_beg_time))
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
    shared_ptr<Pcd2Pgm> p2p;

    /*** keyframe config ***/
    PointCloudType::Ptr feats_undistort;
    PointCloudType::Ptr global_map;

    /*** global map maintain ***/
    string map_path;
    string globalmap_path = PCD_FILE_DIR("globalmap.pcd");
    string trajectory_path = PCD_FILE_DIR("trajectory.pcd");
    string scd_path = PCD_FILE_DIR("scancontext/");
};
