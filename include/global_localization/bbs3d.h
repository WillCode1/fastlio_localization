#pragma once
#include <iostream>
#include <vector>
#include <Eigen/Core>
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

        std::cout << "[YAML] Loading 3D-BBS parameters..." << std::endl;
        min_level_res = conf["min_level_res"].as<double>();
        max_level = conf["max_level"].as<int>();

        if (min_level_res == 0.0 || max_level == 0)
        {
            std::cout << "[ERROR] Set min_level and num_layers except for 0" << std::endl;
            return false;
        }

        std::cout << "[YAML] Loading angular search range..." << std::endl;
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

        std::cout << "[YAML] Loading score threshold percentage..." << std::endl;
        score_threshold_percentage = conf["score_threshold_percentage"].as<double>();

        std::cout << "[YAML] Loading downsample parameters..." << std::endl;
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
        std::cout << "[YAML] Loading 3D-BBS parameters..." << std::endl;
        ros::param::param("min_level_res", min_level_res, 1.0);
        ros::param::param("max_level", max_level, 6);

        if (min_level_res == 0.0 || max_level == 0)
        {
            std::cout << "[ERROR] Set min_level and num_layers except for 0" << std::endl;
            return false;
        }

        std::cout << "[YAML] Loading angular search range..." << std::endl;
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

        std::cout << "[YAML] Loading score threshold percentage..." << std::endl;
        ros::param::param("score_threshold_percentage", score_threshold_percentage, 0.9);

        std::cout << "[YAML] Loading downsample parameters..." << std::endl;
        ros::param::param("tar_leaf_size", tar_leaf_size, 0.1f);
        ros::param::param("src_leaf_size", src_leaf_size, 2.0f);
        ros::param::param("min_scan_range", min_scan_range, 0.0);
        ros::param::param("max_scan_range", max_scan_range, 100.0);

        ros::param::param("timeout_msec", timeout_msec, 0);

        ros::param::param("use_gicp", use_gicp, false);
        return true;
    }
#endif

    int run();

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
