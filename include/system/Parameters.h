#include "system/System.hpp"
#include <ros/ros.h>

inline void load_ros_parameters(bool &path_en, bool &scan_pub_en, bool &dense_pub_en,
                                std::string &lidar_topic, std::string &imu_topic, std::string &gnss_topic, std::string &map_frame, std::string &lidar_frame, std::string &baselink_frame)
{
    ros::param::param("publish/path_en", path_en, false);
    ros::param::param("publish/scan_publish_en", scan_pub_en, false);
    ros::param::param("publish/dense_publish_en", dense_pub_en, false);

    ros::param::param("common/lidar_topic", lidar_topic, std::string("/livox/lidar"));
    ros::param::param("common/imu_topic", imu_topic, std::string("/livox/imu"));
    ros::param::param("common/gnss_topic", gnss_topic, std::string("/gps/fix"));
    ros::param::param("common/map_frame", map_frame, std::string("camera_init"));
    ros::param::param("common/lidar_frame", lidar_frame, std::string("lidar"));
    ros::param::param("common/baselink_frame", baselink_frame, std::string("base_link"));
}

inline void load_parameters(System &slam, int &lidar_type)
{
    double blind, detect_range;
    int n_scans, scan_rate, time_unit;
    vector<double> extrinT;
    vector<double> extrinR;
    V3D extrinT_eigen;
    M3D extrinR_eigen;
    double gyr_cov, acc_cov, b_gyr_cov, b_acc_cov;

    ros::param::param("mapping/gyr_cov", gyr_cov, 0.1);
    ros::param::param("mapping/acc_cov", acc_cov, 0.1);
    ros::param::param("mapping/b_gyr_cov", b_gyr_cov, 0.0001);
    ros::param::param("mapping/b_acc_cov", b_acc_cov, 0.0001);
    ros::param::param("preprocess/blind", blind, 0.01);
    ros::param::param("preprocess/det_range", detect_range, 300.);
    ros::param::param("preprocess/lidar_type", lidar_type, (int)AVIA);
    ros::param::param("preprocess/scan_line", n_scans, 16);
    ros::param::param("preprocess/timestamp_unit", time_unit, (int)US);
    ros::param::param("preprocess/scan_rate", scan_rate, 10);
    ros::param::param("official/map_path", slam.map_path, std::string(""));
    if (slam.map_path.compare("") != 0)
    {
        slam.globalmap_path = slam.map_path + "/globalmap.pcd";
        slam.trajectory_path = slam.map_path + "/trajectory.pcd";
        slam.scd_path = slam.map_path + "/scancontext/";
    }

    ros::param::param("scan_context/lidar_height", slam.relocalization->sc_manager->LIDAR_HEIGHT, 2.0);
    ros::param::param("scan_context/sc_dist_thres", slam.relocalization->sc_manager->SC_DIST_THRES, 0.5);

    if (true)
    {
        ros::param::param("utm_origin/zone", slam.relocalization->utm_origin.zone, std::string("51N"));
        ros::param::param("utm_origin/east", slam.relocalization->utm_origin.east, 0.);
        ros::param::param("utm_origin/north", slam.relocalization->utm_origin.north, 0.);
        ros::param::param("utm_origin/up", slam.relocalization->utm_origin.up, 0.);

        ros::param::param("mapping/extrinsicT_imu2gnss", extrinT, vector<double>());
        ros::param::param("mapping/extrinsicR_imu2gnss", extrinR, vector<double>());
        extrinT_eigen << VEC_FROM_ARRAY(extrinT);
        extrinR_eigen << MAT_FROM_ARRAY(extrinR);
        slam.relocalization->set_extrinsic(extrinT_eigen, extrinR_eigen);

        ros::param::param("relocalization_cfg/algorithm_type", slam.relocalization->algorithm_type, std::string("UNKONW"));

        BnbOptions match_option;
        ros::param::param("bnb3d/linear_xy_window_size", match_option.linear_xy_window_size, 10.);
        ros::param::param("bnb3d/linear_z_window_size", match_option.linear_z_window_size, 1.);
        ros::param::param("bnb3d/angular_search_window", match_option.angular_search_window, 30.);
        ros::param::param("bnb3d/pc_resolutions", match_option.pc_resolutions, vector<double>());
        ros::param::param("bnb3d/bnb_depth", match_option.bnb_depth, 5);
        ros::param::param("bnb3d/min_score", match_option.min_score, 0.1);
        ros::param::param("bnb3d/enough_score", match_option.enough_score, 0.8);
        ros::param::param("bnb3d/min_xy_resolution", match_option.min_xy_resolution, 0.2);
        ros::param::param("bnb3d/min_z_resolution", match_option.min_z_resolution, 0.1);
        ros::param::param("bnb3d/min_angular_resolution", match_option.min_angular_resolution, 0.1);
        ros::param::param("bnb3d/thread_num", match_option.thread_num, 4);
        ros::param::param("bnb3d/filter_size_scan", match_option.filter_size_scan, 0.1);
        ros::param::param("bnb3d/debug_mode", match_option.debug_mode, false);

        Pose lidar_extrinsic;
        ros::param::param("relocalization_cfg/lidar_ext/x", lidar_extrinsic.x, 0.);
        ros::param::param("relocalization_cfg/lidar_ext/y", lidar_extrinsic.y, 0.);
        ros::param::param("relocalization_cfg/lidar_ext/z", lidar_extrinsic.z, 0.);
        ros::param::param("relocalization_cfg/lidar_ext/roll", lidar_extrinsic.roll, 0.);
        ros::param::param("relocalization_cfg/lidar_ext/pitch", lidar_extrinsic.pitch, 0.);
        ros::param::param("relocalization_cfg/lidar_ext/yaw", lidar_extrinsic.yaw, 0.);
        slam.relocalization->set_bnb3d_param(match_option, lidar_extrinsic);

        double step_size, resolution;
        ros::param::param("ndt/step_size", step_size, 0.1);
        ros::param::param("ndt/resolution", resolution, 1.);
        slam.relocalization->set_ndt_param(step_size, resolution);

        bool use_gicp;
        double gicp_downsample, filter_range, search_radius, teps, feps, fitness_score;
        ros::param::param("gicp/use_gicp", use_gicp, true);
        ros::param::param("gicp/filter_range", filter_range, 80.);
        ros::param::param("gicp/gicp_downsample", gicp_downsample, 0.2);
        ros::param::param("gicp/search_radius", search_radius, 0.5);
        ros::param::param("gicp/teps", teps, 1e-3);
        ros::param::param("gicp/feps", feps, 1e-3);
        ros::param::param("gicp/fitness_score", fitness_score, 0.3);
        slam.relocalization->set_gicp_param(use_gicp, filter_range, gicp_downsample, search_radius, teps, feps, fitness_score);
    }

    int frontend_type;
    ros::param::param("mapping/frontend_type", frontend_type, 0);
    if (frontend_type == Fastlio)
    {
        slam.frontend = make_shared<FastlioOdometry>();
        LOG_WARN("frontend use fastlio!");
    }
    else if (frontend_type == Pointlio)
    {
        double acc_cov_output, gyr_cov_output, gyr_cov_input, acc_cov_input, vel_cov, imu_meas_acc_cov, imu_meas_omg_cov;
        slam.frontend = make_shared<PointlioOdometry>();
        LOG_WARN("frontend use pointlio!");
        auto pointlio = dynamic_cast<PointlioOdometry *>(slam.frontend.get());
        ros::param::param("mapping/imu_en", pointlio->imu_en, true);
        ros::param::param("mapping/use_imu_as_input", pointlio->use_imu_as_input, true);
        ros::param::param("mapping/prop_at_freq_of_imu", pointlio->prop_at_freq_of_imu, true);
        ros::param::param("mapping/check_saturation", pointlio->check_saturation, true);
        ros::param::param("mapping/saturation_acc", pointlio->saturation_acc, 3.0);
        ros::param::param("mapping/saturation_gyro", pointlio->saturation_gyro, 35.0);
        ros::param::param("mapping/acc_cov_output", acc_cov_output, 500.);
        ros::param::param("mapping/gyr_cov_output", gyr_cov_output, 1000.);
        ros::param::param("mapping/gyr_cov_input", gyr_cov_input, 0.01);
        ros::param::param("mapping/acc_cov_input", acc_cov_input, 0.1);
        ros::param::param("mapping/vel_cov", vel_cov, 20.);
        pointlio->Q_input = process_noise_cov_input(gyr_cov_input, acc_cov_input, b_gyr_cov, b_acc_cov);
        pointlio->Q_output = process_noise_cov_output(vel_cov, gyr_cov_output, acc_cov_output, b_gyr_cov, b_acc_cov);
        ros::param::param("mapping/imu_meas_acc_cov", imu_meas_acc_cov, 0.1);
        ros::param::param("mapping/imu_meas_omg_cov", imu_meas_omg_cov, 0.1);
        pointlio->R_imu << imu_meas_omg_cov, imu_meas_omg_cov, imu_meas_omg_cov, imu_meas_acc_cov, imu_meas_acc_cov, imu_meas_acc_cov;
    }
    else
    {
        LOG_ERROR("frontend odom type error!");
        exit(100);
    }

    vector<int> valid_ring_index;
    valid_ring_index = config["mapping"]["valid_ring"].IsDefined() ? config["mapping"]["valid_ring"].as<vector<int>>() : vector<int>();
    slam.frontend->lidar->valid_ring.insert(valid_ring_index.begin(), valid_ring_index.end());
    slam.frontend->lidar->init(n_scans, scan_rate, time_unit, blind, detect_range);
    ros::param::param("preprocess/imu_rate", slam.frontend->imu->imu_rate, 200);
    ros::param::param("preprocess/imu_meas_check", slam.frontend->imu->imu_meas_check, vector<double>(6, 20));
    slam.frontend->imu->set_imu_cov(process_noise_cov(gyr_cov, acc_cov, b_gyr_cov, b_acc_cov));

    ros::param::param("common/timedelay_lidar2imu", slam.frontend->timedelay_lidar2imu, 0.);
    ros::param::param("mapping/gravity_init", slam.frontend->gravity_init, vector<double>());

    ros::param::param("mapping/max_iteration", slam.frontend->num_max_iterations, 4);
    ros::param::param("mapping/surf_frame_ds_res", slam.frontend->surf_frame_ds_res, 0.5);
    ros::param::param("mapping/point_skip_num", slam.frontend->point_skip_num, 2);
    ros::param::param("mapping/space_down_sample", slam.frontend->space_down_sample, true);
    ros::param::param("mapping/ikdtree_resolution", slam.frontend->ikdtree_resolution, 0.5);
    ros::param::param("mapping/lidar_model_search_range", slam.frontend->lidar_model_search_range, 5.);
    ros::param::param("mapping/lidar_meas_cov", slam.frontend->lidar_meas_cov, 0.001);
    ros::param::param("mapping/cube_len", slam.frontend->cube_len, 200.);
    ros::param::param("mapping/extrinsic_est_en", slam.frontend->extrinsic_est_en, true);
    ros::param::param("mapping/runtime_log_enable", slam.frontend->loger.runtime_log, 0);

    ros::param::param("mapping/extrinsic_T", extrinT, vector<double>());
    ros::param::param("mapping/extrinsic_R", extrinR, vector<double>());
    extrinT_eigen << VEC_FROM_ARRAY(extrinT);
    extrinR_eigen << MAT_FROM_ARRAY(extrinR);
    slam.frontend->set_extrinsic(extrinT_eigen, extrinR_eigen);

    slam.init_system_mode();
}

inline void load_log_parameters(bool &relocate_use_last_pose, std::string &last_pose_record_path, bool &location_log_enable, std::string &location_log_save_path)
{
	ros::param::param("official/relocate_use_last_pose", relocate_use_last_pose, true);
	ros::param::param("official/last_pose_record_path", last_pose_record_path, std::string(""));
    ros::param::param("official/location_log_enable", location_log_enable, false);
    ros::param::param("official/location_log_save_path", location_log_save_path, std::string(""));
}
