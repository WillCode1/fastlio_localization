#pragma once
#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <cpu_bbs3d/bbs3d.hpp>
#include <gpu_bbs3d/bbs3d.cuh>
#include <pointcloud_iof/pcl_eigen_converter.hpp>
// #include <pointcloud_iof/pcd_loader.hpp>
#if 0
#include <yaml-cpp/yaml.h>
#endif
#if 1
#include <ros/ros.h>
#endif

class BBS3D
{
public:
#if 0
    bool load_config(const std::string &config)
    {
        YAML::Node conf = YAML::LoadFile(config);

        std::cout << "[BBS3D] Loading 3D-BBS parameters..." << std::endl;
        min_level_res = conf["min_level_res"].as<double>();
        max_level = conf["max_level"].as<int>();

        if (min_level_res == 0.0 || max_level == 0)
        {
            std::cout << "[ERROR] Set min_level and num_layers except for 0" << std::endl;
            return false;
        }

        std::cout << "[BBS3D] Loading angular search range..." << std::endl;
        std::vector<double> min_rpy_temp = conf["min_rpy"].as<std::vector<double>>();
        std::vector<double> max_rpy_temp = conf["max_rpy"].as<std::vector<double>>();
        if (min_rpy_temp.size() == 3 && max_rpy_temp.size() == 3)
        {
            min_rpy = to_eigen(min_rpy_temp);
            max_rpy = to_eigen(max_rpy_temp);
        }
        else
        {
            std::cout << "[ERROR] Set min_rpy and max_rpy correctly" << std::endl;
            return false;
        }

        std::cout << "[BBS3D] Loading score threshold percentage..." << std::endl;
        score_threshold_percentage = conf["score_threshold_percentage"].as<double>();

        std::cout << "[BBS3D] Loading downsample parameters..." << std::endl;
        tar_leaf_size = conf["tar_leaf_size"].as<float>();
        src_leaf_size = conf["src_leaf_size"].as<float>();
        min_scan_range = conf["min_scan_range"].as<double>();
        max_scan_range = conf["max_scan_range"].as<double>();

        timeout_msec = conf["timeout_msec"].as<int>();

        use_gicp = conf["use_gicp"].as<bool>();
        return true;
    }
#endif

#if 1
    bool load_config()
    {
        ros::param::param("target_clouds", tar_path, std::string(""));

        std::cout << "[BBS3D] Loading 3D-BBS parameters..." << std::endl;
        ros::param::param("min_level_res", min_level_res, 1.0);
        ros::param::param("max_level", max_level, 6);

        if (min_level_res == 0.0 || max_level == 0)
        {
            std::cout << "[ERROR] Set min_level and num_layers except for 0" << std::endl;
            return false;
        }

        std::cout << "[BBS3D] Loading angular search range..." << std::endl;
        std::vector<double> min_rpy_temp;
        std::vector<double> max_rpy_temp;
        ros::param::param("min_rpy", min_rpy_temp, vector<double>());
        ros::param::param("max_rpy", max_rpy_temp, vector<double>());
        if (min_rpy_temp.size() == 3 && max_rpy_temp.size() == 3)
        {
            min_rpy = to_eigen(min_rpy_temp);
            max_rpy = to_eigen(max_rpy_temp);
        }
        else
        {
            std::cout << "[ERROR] Set min_rpy and max_rpy correctly" << std::endl;
            return false;
        }

        std::cout << "[BBS3D] Loading score threshold percentage..." << std::endl;
        ros::param::param("score_threshold_percentage", score_threshold_percentage, 0.9);

        std::cout << "[BBS3D] Loading downsample parameters..." << std::endl;
        ros::param::param("tar_leaf_size", tar_leaf_size, 0.1f);
        ros::param::param("src_leaf_size", src_leaf_size, 2.0f);
        ros::param::param("min_scan_range", min_scan_range, 0.0);
        ros::param::param("max_scan_range", max_scan_range, 100.0);

        ros::param::param("timeout_msec", timeout_msec, 0);

        ros::param::param("use_gicp", use_gicp, false);

        // ====3D-BBS====
#ifdef USE_CUDA
        bbs3d_ptr = std::make_unique<gpu::BBS3D>();
#else
        bbs3d_ptr = std::make_unique<cpu::BBS3D>();
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
            std::cout << "[Voxel map] Loaded voxelmaps failed!" << std::endl;
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
        int num_threads = 4;
        bbs3d_ptr->set_num_threads(num_threads);
#endif
        return true;
    }
#endif

    bool run(pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud)
    {
        double sum_time = 0;
#ifdef USE_CUDA
        std::vector<Eigen::Vector3f> src_points;
#else
        std::vector<Eigen::Vector3d> src_points;
#endif
        pciof::pcl_to_eigen(src_cloud, src_points);
        bbs3d_ptr->set_src_points(src_points);
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

        sum_time = bbs3d_ptr->get_elapsed_time();
        std::cout << "[Localize] Average time: " << sum_time << "[msec] per frame" << std::endl;
        return true;
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

private:
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
