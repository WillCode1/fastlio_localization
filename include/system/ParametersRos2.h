#include "system/System.hpp"
#include <rclcpp/rclcpp.hpp>

inline void load_ros_parameters(rclcpp::Node::SharedPtr &node, bool &path_en, bool &scan_pub_en, bool &dense_pub_en, bool &lidar_tf_broadcast, bool &imu_tf_broadcast,
                                std::string &lidar_topic, std::string &imu_topic, std::string &gnss_topic, std::string &map_frame, std::string &lidar_frame, std::string &baselink_frame)
{
    node->declare_parameter("path_en", false);
    node->declare_parameter("scan_publish_en", false);
    node->declare_parameter("dense_publish_en", false);
    node->declare_parameter("lidar_tf_broadcast", false);
    node->declare_parameter("imu_tf_broadcast", false);

    node->declare_parameter("lidar_topic", "/livox/lidar");
    node->declare_parameter("imu_topic", "/livox/imu");
    node->declare_parameter("gnss_topic", "/gps/fix");
    node->declare_parameter("map_frame", "camera_init");
    node->declare_parameter("lidar_frame", "lidar");
    node->declare_parameter("baselink_frame", "base_link");

    node->get_parameter("path_en", path_en);
    node->get_parameter("scan_publish_en", scan_pub_en);
    node->get_parameter("dense_publish_en", dense_pub_en);
    node->get_parameter("lidar_tf_broadcast", lidar_tf_broadcast);
    node->get_parameter("imu_tf_broadcast", imu_tf_broadcast);

    node->get_parameter("lidar_topic", lidar_topic);
    node->get_parameter("imu_topic", imu_topic);
    node->get_parameter("gnss_topic", gnss_topic);
    node->get_parameter("map_frame", map_frame);
    node->get_parameter("lidar_frame", lidar_frame);
    node->get_parameter("baselink_frame", baselink_frame);
}

inline void load_parameters(rclcpp::Node::SharedPtr &node, System &slam, int &lidar_type)
{
    double blind, detect_range;
    int n_scans, scan_rate, time_unit;
    vector<double> extrinT;
    vector<double> extrinR;
    V3D extrinT_eigen;
    M3D extrinR_eigen;
    double gyr_cov, acc_cov, b_gyr_cov, b_acc_cov;

    node->declare_parameter("gyr_cov", 0.1);
    node->declare_parameter("acc_cov", 0.1);
    node->declare_parameter("b_gyr_cov", 0.0001);
    node->declare_parameter("b_acc_cov", 0.0001);
    node->declare_parameter("blind", 0.01);
    node->declare_parameter("det_range", 300.);
    node->declare_parameter("lidar_type", 1);
    node->declare_parameter("scan_line", 16);
    node->declare_parameter("timestamp_unit", 2);
    node->declare_parameter("scan_rate", 10);
    node->declare_parameter("map_path", "");
    node->declare_parameter("lidar_height", 2.0);
    node->declare_parameter("sc_dist_thres", 0.5);
    node->declare_parameter("extrinsic_T", vector<double>());
    node->declare_parameter("extrinsic_R", vector<double>());

    node->get_parameter("gyr_cov", gyr_cov);
    node->get_parameter("acc_cov", acc_cov);
    node->get_parameter("b_gyr_cov", b_gyr_cov);
    node->get_parameter("b_acc_cov", b_acc_cov);
    node->get_parameter("blind", blind);
    node->get_parameter("det_range", detect_range);
    node->get_parameter("lidar_type", lidar_type);
    node->get_parameter("scan_line", n_scans);
    node->get_parameter("timestamp_unit", time_unit);
    node->get_parameter("scan_rate", scan_rate);
    node->get_parameter("map_path", slam.map_path);
    if (slam.map_path.compare("") != 0)
    {
        slam.globalmap_path = slam.map_path + "/globalmap.pcd";
        slam.trajectory_path = slam.map_path + "/trajectory.pcd";
        slam.scd_path = slam.map_path + "/scancontext/";
    }

    node->get_parameter("lidar_height", slam.relocalization->sc_manager->LIDAR_HEIGHT);
    node->get_parameter("sc_dist_thres", slam.relocalization->sc_manager->SC_DIST_THRES);

    if (true)
    {
        node->declare_parameter("extrinsicT_imu2gnss", vector<double>());
        node->declare_parameter("extrinsicR_imu2gnss", vector<double>());
        node->declare_parameter("relocal_cfg_algorithm_type", "UNKONW");

        node->get_parameter("extrinsicT_imu2gnss", extrinT);
        node->get_parameter("extrinsicR_imu2gnss", extrinR);
        extrinT_eigen << VEC_FROM_ARRAY(extrinT);
        extrinR_eigen << MAT_FROM_ARRAY(extrinR);
        slam.relocalization->set_extrinsic(extrinT_eigen, extrinR_eigen);

        node->get_parameter("relocal_cfg_algorithm_type", slam.relocalization->algorithm_type);

        BnbOptions match_option;
        node->declare_parameter("bnb3d_linear_xy_window_size", 10.);
        node->declare_parameter("bnb3d_linear_z_window_size", 1.);
        node->declare_parameter("bnb3d_angular_search_window", 30.);
        node->declare_parameter("bnb3d_pc_resolutions", vector<double>());
        node->declare_parameter("bnb3d_bnb_depth", 5);
        node->declare_parameter("bnb3d_min_score", 0.1);
        node->declare_parameter("bnb3d_enough_score", 0.8);
        node->declare_parameter("bnb3d_min_xy_resolution", 0.2);
        node->declare_parameter("bnb3d_min_z_resolution", 0.1);
        node->declare_parameter("bnb3d_min_angular_resolution", 0.1);
        node->declare_parameter("bnb3d_filter_size_scan", 0.1);
        node->declare_parameter("bnb3d_debug_mode", false);

        node->get_parameter("bnb3d_linear_xy_window_size", match_option.linear_xy_window_size);
        node->get_parameter("bnb3d_linear_z_window_size", match_option.linear_z_window_size);
        node->get_parameter("bnb3d_angular_search_window", match_option.angular_search_window);
        node->get_parameter("bnb3d_pc_resolutions", match_option.pc_resolutions);
        node->get_parameter("bnb3d_depth", match_option.bnb_depth);
        node->get_parameter("bnb3d_min_score", match_option.min_score);
        node->get_parameter("bnb3d_enough_score", match_option.enough_score);
        node->get_parameter("bnb3d_min_xy_resolution", match_option.min_xy_resolution);
        node->get_parameter("bnb3d_min_z_resolution", match_option.min_z_resolution);
        node->get_parameter("bnb3d_min_angular_resolution", match_option.min_angular_resolution);
        node->get_parameter("bnb3d_filter_size_scan", match_option.filter_size_scan);
        node->get_parameter("bnb3d_debug_mode", match_option.debug_mode);

        node->get_parameter("extrinsic_T", extrinT);
        node->get_parameter("extrinsic_R", extrinR);
        extrinT_eigen << VEC_FROM_ARRAY(extrinT);
        extrinR_eigen << MAT_FROM_ARRAY(extrinR);
        V3D ext_rpy = EigenMath::RotationMatrix2RPY(extrinR_eigen);
        Pose lidar_extrinsic;
        lidar_extrinsic.x = extrinT_eigen.x();
        lidar_extrinsic.y = extrinT_eigen.y();
        lidar_extrinsic.z = extrinT_eigen.z();
        lidar_extrinsic.roll = ext_rpy.x();
        lidar_extrinsic.pitch = ext_rpy.y();
        lidar_extrinsic.yaw = ext_rpy.z();
        slam.relocalization->set_bnb3d_param(match_option, lidar_extrinsic);

        double step_size, resolution;
        node->declare_parameter("ndt_step_size", 0.1);
        node->declare_parameter("ndt_resolution", 1.);
        node->get_parameter("ndt_step_size", step_size);
        node->get_parameter("ndt_resolution", resolution);
        slam.relocalization->set_ndt_param(step_size, resolution);

        bool use_gicp;
        double gicp_downsample, filter_range, search_radius, teps, feps, fitness_score;
        node->declare_parameter("gicp_use_gicp", true);
        node->declare_parameter("gicp_filter_range", 80.);
        node->declare_parameter("gicp_downsample", 0.2);
        node->declare_parameter("gicp_search_radius", 0.5);
        node->declare_parameter("gicp_teps", 1e-3);
        node->declare_parameter("gicp_feps", 1e-3);
        node->declare_parameter("gicp_fitness_score", 0.3);

        node->get_parameter("gicp_use_gicp", use_gicp);
        node->get_parameter("gicp_filter_range", filter_range);
        node->get_parameter("gicp_gicp_downsample", gicp_downsample);
        node->get_parameter("gicp_search_radius", search_radius);
        node->get_parameter("gicp_teps", teps);
        node->get_parameter("gicp_feps", feps);
        node->get_parameter("gicp_fitness_score", fitness_score);
        slam.relocalization->set_gicp_param(use_gicp, filter_range, gicp_downsample, search_radius, teps, feps, fitness_score);
    }

    int frontend_type;
    node->declare_parameter("frontend_type", 0);
    node->get_parameter("frontend_type", frontend_type);
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
        node->declare_parameter("imu_en", true);
        node->declare_parameter("use_imu_as_input", true);
        node->declare_parameter("prop_at_freq_of_imu", true);
        node->declare_parameter("check_saturation", true);
        node->declare_parameter("saturation_acc", 3.0);
        node->declare_parameter("saturation_gyro", 35.0);
        node->declare_parameter("acc_cov_output", 500.);
        node->declare_parameter("gyr_cov_output", 1000.);
        node->declare_parameter("gyr_cov_input", 0.01);
        node->declare_parameter("acc_cov_input", 0.1);
        node->declare_parameter("vel_cov", 20.);
        node->declare_parameter("imu_meas_acc_cov", 0.1);
        node->declare_parameter("imu_meas_omg_cov", 0.1);

        node->get_parameter("imu_en", pointlio->imu_en);
        node->get_parameter("use_imu_as_input", pointlio->use_imu_as_input);
        node->get_parameter("prop_at_freq_of_imu", pointlio->prop_at_freq_of_imu);
        node->get_parameter("check_saturation", pointlio->check_saturation);
        node->get_parameter("saturation_acc", pointlio->saturation_acc);
        node->get_parameter("saturation_gyro", pointlio->saturation_gyro);
        node->get_parameter("acc_cov_output", acc_cov_output);
        node->get_parameter("gyr_cov_output", gyr_cov_output);
        node->get_parameter("gyr_cov_input", gyr_cov_input);
        node->get_parameter("acc_cov_input", acc_cov_input);
        node->get_parameter("vel_cov", vel_cov);
        pointlio->Q_input = process_noise_cov_input(gyr_cov_input, acc_cov_input, b_gyr_cov, b_acc_cov);
        pointlio->Q_output = process_noise_cov_output(vel_cov, gyr_cov_output, acc_cov_output, b_gyr_cov, b_acc_cov);
        node->get_parameter("imu_meas_acc_cov", imu_meas_acc_cov);
        node->get_parameter("imu_meas_omg_cov", imu_meas_omg_cov);
        pointlio->R_imu << imu_meas_omg_cov, imu_meas_omg_cov, imu_meas_omg_cov, imu_meas_acc_cov, imu_meas_acc_cov, imu_meas_acc_cov;
    }
    else
    {
        LOG_ERROR("frontend odom type error!");
        exit(100);
    }

    // node->declare_parameter("valid_ring", vector<int>());
    node->declare_parameter("imu_rate", 200);
    node->declare_parameter("imu_meas_check", vector<double>(6, 20));
    node->declare_parameter("timedelay_lidar2imu", 0.);
    node->declare_parameter("gravity_init", vector<double>());
    node->declare_parameter("max_iteration", 4);
    node->declare_parameter("surf_frame_ds_res", 0.5);
    node->declare_parameter("point_skip_num", 2);
    node->declare_parameter("space_down_sample", true);
    node->declare_parameter("ikdtree_resolution", 0.5);
    node->declare_parameter("lidar_model_search_range", 5.);
    node->declare_parameter("lidar_meas_cov", 0.001);
    node->declare_parameter("cube_len", 200.);
    node->declare_parameter("extrinsic_est_en", true);
    node->declare_parameter("runtime_log_enable", 0);

    // vector<int> valid_ring_index;
    // node->get_parameter("valid_ring", valid_ring_index);
    // slam.frontend->lidar->valid_ring.insert(valid_ring_index.begin(), valid_ring_index.end());
    slam.frontend->lidar->init(n_scans, scan_rate, time_unit, blind, detect_range);
    node->get_parameter("imu_rate", slam.frontend->imu->imu_rate);
    node->get_parameter("imu_meas_check", slam.frontend->imu->imu_meas_check);
    slam.frontend->imu->set_imu_cov(process_noise_cov(gyr_cov, acc_cov, b_gyr_cov, b_acc_cov));

    node->get_parameter("timedelay_lidar2imu", slam.frontend->timedelay_lidar2imu);
    node->get_parameter("gravity_init", slam.frontend->gravity_init);

    node->get_parameter("max_iteration", slam.frontend->num_max_iterations);
    node->get_parameter("surf_frame_ds_res", slam.frontend->surf_frame_ds_res);
    node->get_parameter("point_skip_num", slam.frontend->point_skip_num);
    node->get_parameter("space_down_sample", slam.frontend->space_down_sample);
    node->get_parameter("ikdtree_resolution", slam.frontend->ikdtree_resolution);
    node->get_parameter("lidar_model_search_range", slam.frontend->lidar_model_search_range);
    node->get_parameter("lidar_meas_cov", slam.frontend->lidar_meas_cov);
    node->get_parameter("cube_len", slam.frontend->cube_len);
    node->get_parameter("extrinsic_est_en", slam.frontend->extrinsic_est_en);
    node->get_parameter("runtime_log_enable", slam.frontend->loger.runtime_log);

    node->get_parameter("extrinsic_T", extrinT);
    node->get_parameter("extrinsic_R", extrinR);
    extrinT_eigen << VEC_FROM_ARRAY(extrinT);
    extrinR_eigen << MAT_FROM_ARRAY(extrinR);
    slam.frontend->set_extrinsic(extrinT_eigen, extrinR_eigen);

    slam.init_system_mode();
}

inline void load_log_parameters(rclcpp::Node::SharedPtr &node, bool &relocate_use_last_pose, std::string &last_pose_record_path, bool &location_log_enable, std::string &location_log_save_path)
{
    node->declare_parameter("relocate_use_last_pose", true);
    node->declare_parameter("last_pose_record_path", "");
    node->declare_parameter("location_log_enable", false);
    node->declare_parameter("location_log_save_path", "");

    node->get_parameter("relocate_use_last_pose", relocate_use_last_pose);
    node->get_parameter("last_pose_record_path", last_pose_record_path);
    node->get_parameter("location_log_enable", location_log_enable);
    node->get_parameter("location_log_save_path", location_log_save_path);
}
