#pragma once
#include <iostream>
#include <vector>
#include <Eigen/Core>
#ifdef USE_CUDA
#include "global_localization/bbs3d/gpu_bbs3d/bbs3d.cuh"
#else
#include "global_localization/bbs3d/cpu_bbs3d/bbs3d.hpp"
#endif
#include <pointcloud_iof/pcl_eigen_converter.hpp>
// #include <pointcloud_iof/pcd_loader.hpp>

struct BBS3DOptions
{
    std::string tar_path;

    // 3D-BBS parameters
    double min_level_res;
    int max_level;

    // angular search range
    std::vector<double> min_rpy;
    std::vector<double> max_rpy;

    // score threshold percentage
    double score_threshold_percentage;

    int num_threads;

    // downsample
    double tar_leaf_size, src_leaf_size;
    double min_scan_range, max_scan_range;

    // timeout
    int timeout_msec;

    // align
    bool use_gicp;
};

class BBS3D
{
public:
    bool load_config(const BBS3DOptions &bbs3d_option)
    {
        tar_path = bbs3d_option.tar_path;

        std::cout << "[BBS3D] Loading 3D-BBS parameters..." << std::endl;
        min_level_res = bbs3d_option.min_level_res;
        max_level = bbs3d_option.max_level;

        if (min_level_res == 0.0 || max_level == 0)
        {
            std::cout << "[ERROR] Set min_level and num_layers except for 0" << std::endl;
            return false;
        }

        std::cout << "[BBS3D] Loading angular search range..." << std::endl;
        if (bbs3d_option.min_rpy.size() == 3 && bbs3d_option.max_rpy.size() == 3)
        {
            min_rpy = to_eigen(bbs3d_option.min_rpy);
            max_rpy = to_eigen(bbs3d_option.max_rpy);
        }
        else
        {
            std::cout << "[ERROR] Set min_rpy and max_rpy correctly" << std::endl;
            return false;
        }

        std::cout << "[BBS3D] Loading score threshold percentage..." << std::endl;
        score_threshold_percentage = bbs3d_option.score_threshold_percentage;

        std::cout << "[BBS3D] Loading downsample parameters..." << std::endl;
        tar_leaf_size = bbs3d_option.tar_leaf_size;
        src_leaf_size = bbs3d_option.src_leaf_size;
        min_scan_range = bbs3d_option.min_scan_range;
        max_scan_range = bbs3d_option.max_scan_range;

        timeout_msec = bbs3d_option.timeout_msec;
        use_gicp = bbs3d_option.use_gicp;

        // ====3D-BBS====
#ifdef USE_CUDA
        bbs3d_ptr = std::make_unique<gpu::BBS3D>();
        std::cout << "[Localize] USE CUDA!" << std::endl;
#else
        bbs3d_ptr = std::make_unique<cpu::BBS3D>();
        std::cout << "[Localize] NOT USE CUDA!" << std::endl;
#endif

        // Set target points
        std::cout << "[Voxel map] Creating hierarchical voxel map..." << std::endl;
        auto initi_t1 = std::chrono::high_resolution_clock::now();
        if (bbs3d_ptr->set_voxelmaps_coords(tar_path))
        {
            std::cout << "[Voxel map] Loaded voxelmaps coords directly" << std::endl;
        }
        else
        {
            // bbs3d_ptr->set_tar_points(tar_points, min_level_res, max_level);
            // bbs3d_ptr->set_trans_search_range(tar_points);
            std::cout << "\033[31m" << "[Voxel map] Loaded voxelmaps failed!" << "\033[0m" << std::endl;
            return false;
        }
        auto init_t2 = std::chrono::high_resolution_clock::now();
        double init_time = std::chrono::duration_cast<std::chrono::nanoseconds>(init_t2 - initi_t1).count() / 1e6;
        std::cout << "[Voxel map] Execution time: " << init_time << "[msec] " << std::endl;

#ifdef USE_CUDA
        bbs3d_ptr->set_angular_search_range(min_rpy.cast<float>(), max_rpy.cast<float>());
        bbs3d_ptr->set_score_threshold_percentage(static_cast<float>(score_threshold_percentage));
#else
        bbs3d_ptr->set_angular_search_range(min_rpy, max_rpy);
        bbs3d_ptr->set_score_threshold_percentage(score_threshold_percentage);
#endif
        if (timeout_msec > 0)
        {
            bbs3d_ptr->enable_timeout();
            bbs3d_ptr->set_timeout_duration_in_msec(timeout_msec);
        }

#ifndef USE_CUDA
        bbs3d_ptr->set_num_threads(bbs3d_option.num_threads);
#endif
        return true;
    }

    bool run(pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud, Eigen::Matrix4d &lidar_pose_mat)
    {
#ifdef USE_CUDA
        std::vector<Eigen::Vector3f> src_points;
#else
        std::vector<Eigen::Vector3d> src_points;
#endif
        pciof::pcl_to_eigen(src_cloud, src_points);
        bbs3d_ptr->set_src_points(src_points);
        std::cout << "[Localize] bbs3d start localize." << std::endl;
        bbs3d_ptr->localize();

        std::cout << "[Localize] Execution time: " << bbs3d_ptr->get_elapsed_time() << "[msec] " << std::endl;
        std::cout << "[Localize] Score: " << bbs3d_ptr->get_best_score() << std::endl;

        if (!bbs3d_ptr->has_localized())
        {
            if (bbs3d_ptr->has_timed_out())
                std::cout << "[Failed] Localization timed out." << std::endl;
            else
                std::cout << "[Failed] Score is below the threshold." << std::endl;
            return false;
        }

        lidar_pose_mat = bbs3d_ptr->get_global_pose().cast<double>();
        return true;
    }

    void reset_rpy(const Eigen::Vector3d &rpy_fix)
    {
        auto tmp_min_rpy = min_rpy + rpy_fix;
        auto tmp_max_rpy = max_rpy + rpy_fix;

#ifdef USE_CUDA
        bbs3d_ptr->set_angular_search_range(tmp_min_rpy.cast<float>(), tmp_max_rpy.cast<float>());
#else
        bbs3d_ptr->set_angular_search_range(tmp_min_rpy, tmp_max_rpy);
#endif
    }

private:
    Eigen::Vector3d to_eigen(const std::vector<double>& vec) {
        Eigen::Vector3d e_vec;
        for (int i = 0; i < 3; ++i) {
            if (vec[i] == 6.28) {
            e_vec(i) = 2 * M_PI;
            } else {
            e_vec(i) = vec[i];
            }
        }
        return e_vec;
    }

public:
#ifdef USE_CUDA
    std::unique_ptr<gpu::BBS3D> bbs3d_ptr;
#else
    std::unique_ptr<cpu::BBS3D> bbs3d_ptr;
#endif

    std::string tar_path;

    // 3D-BBS parameters
    double min_level_res;
    int max_level;

    // angular search range
    Eigen::Vector3d min_rpy;
    Eigen::Vector3d max_rpy;

    // score threshold percentage
    double score_threshold_percentage;

    // downsample
    float tar_leaf_size, src_leaf_size;
    double min_scan_range, max_scan_range;

    // timeout
    int timeout_msec;

    // align
    bool use_gicp;
};
