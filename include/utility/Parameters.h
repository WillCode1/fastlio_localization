#include <yaml-cpp/yaml.h>
#include "system/System.hpp"

inline void load_ros_parameters(const std::string &config_path, bool &path_en, bool &scan_pub_en, bool &dense_pub_en,
                                std::string &lidar_topic, std::string &imu_topic, std::string &map_frame, std::string &lidar_frame, std::string &baselink_frame)
{
    YAML::Node config = YAML::LoadFile(config_path);
    path_en = config["publish"]["path_en"].IsDefined() ? config["publish"]["path_en"].as<bool>() : false;
    scan_pub_en = config["publish"]["scan_publish_en"].IsDefined() ? config["publish"]["scan_publish_en"].as<bool>() : false;
    dense_pub_en = config["publish"]["dense_publish_en"].IsDefined() ? config["publish"]["dense_publish_en"].as<bool>() : false;

    lidar_topic = config["common"]["lidar_topic"].IsDefined() ? config["common"]["lidar_topic"].as<string>() : std::string("/livox/lidar");
    imu_topic = config["common"]["imu_topic"].IsDefined() ? config["common"]["imu_topic"].as<string>() : std::string("/livox/imu");
    map_frame = config["common"]["map_frame"].IsDefined() ? config["common"]["map_frame"].as<string>() : std::string("camera_init");
    lidar_frame = config["common"]["lidar_frame"].IsDefined() ? config["common"]["lidar_frame"].as<string>() : std::string("lidar");
    baselink_frame = config["common"]["baselink_frame"].IsDefined() ? config["common"]["baselink_frame"].as<string>() : std::string("baselink");
}

inline void load_parameters(System &slam, const std::string &config_path, int &lidar_type)
{
    YAML::Node config = YAML::LoadFile(config_path);

    double blind, detect_range;
    int n_scans, scan_rate, time_unit;
    vector<double> extrinT;
    vector<double> extrinR;
    V3D extrinT_eigen;
    M3D extrinR_eigen;
    double gyr_cov, acc_cov, b_gyr_cov, b_acc_cov;

    gyr_cov = config["mapping"]["gyr_cov"].IsDefined() ? config["mapping"]["gyr_cov"].as<double>() : 0.1;
    acc_cov = config["mapping"]["acc_cov"].IsDefined() ? config["mapping"]["acc_cov"].as<double>() : 0.1;
    b_gyr_cov = config["mapping"]["b_gyr_cov"].IsDefined() ? config["mapping"]["b_gyr_cov"].as<double>() : 0.0001;
    b_acc_cov = config["mapping"]["b_acc_cov"].IsDefined() ? config["mapping"]["b_acc_cov"].as<double>() : 0.0001;
    blind = config["preprocess"]["blind"].IsDefined() ? config["preprocess"]["blind"].as<double>() : 0.01;
    detect_range = config["preprocess"]["det_range"].IsDefined() ? config["preprocess"]["det_range"].as<double>() : 300.f;
    lidar_type = config["preprocess"]["lidar_type"].IsDefined() ? config["preprocess"]["lidar_type"].as<int>() : AVIA;
    n_scans = config["preprocess"]["scan_line"].IsDefined() ? config["preprocess"]["scan_line"].as<int>() : 16;
    time_unit = config["preprocess"]["timestamp_unit"].IsDefined() ? config["preprocess"]["timestamp_unit"].as<int>() : US;
    scan_rate = config["preprocess"]["scan_rate"].IsDefined() ? config["preprocess"]["scan_rate"].as<int>() : 10;

    slam.relocalization->sc_manager->LIDAR_HEIGHT = config["scan_context"]["lidar_height"].IsDefined() ? config["scan_context"]["lidar_height"].as<double>() : 2.0;
    slam.relocalization->sc_manager->SC_DIST_THRES = config["scan_context"]["sc_dist_thres"].IsDefined() ? config["scan_context"]["sc_dist_thres"].as<double>() : 0.5;

    if (true)
    {
        slam.relocalization->algorithm_type = config["relocalization_cfg"]["algorithm_type"].IsDefined() ? config["relocalization_cfg"]["algorithm_type"].as<string>() : std::string("UNKONW");

        BnbOptions match_option;
        match_option.linear_xy_window_size = config["bnb3d"]["linear_xy_window_size"].IsDefined() ? config["bnb3d"]["linear_xy_window_size"].as<double>() : 10;
        match_option.linear_z_window_size = config["bnb3d"]["linear_z_window_size"].IsDefined() ? config["bnb3d"]["linear_z_window_size"].as<double>() : 1.;
        match_option.angular_search_window = config["bnb3d"]["angular_search_window"].IsDefined() ? config["bnb3d"]["angular_search_window"].as<double>() : 30;
        match_option.pc_resolutions = config["bnb3d"]["pc_resolutions"].IsDefined() ? config["bnb3d"]["pc_resolutions"].as<vector<double>>() : vector<double>();
        match_option.bnb_depth = config["bnb3d"]["bnb_depth"].IsDefined() ? config["bnb3d"]["bnb_depth"].as<int>() : 5;
        match_option.min_score = config["bnb3d"]["min_score"].IsDefined() ? config["bnb3d"]["min_score"].as<double>() : 0.1;
        match_option.enough_score = config["bnb3d"]["enough_score"].IsDefined() ? config["bnb3d"]["enough_score"].as<double>() : 0.8;
        match_option.min_xy_resolution = config["bnb3d"]["min_xy_resolution"].IsDefined() ? config["bnb3d"]["min_xy_resolution"].as<double>() : 0.2;
        match_option.min_z_resolution = config["bnb3d"]["min_z_resolution"].IsDefined() ? config["bnb3d"]["min_z_resolution"].as<double>() : 0.1;
        match_option.min_angular_resolution = config["bnb3d"]["min_angular_resolution"].IsDefined() ? config["bnb3d"]["min_angular_resolution"].as<double>() : 0.1;
        match_option.thread_num = config["bnb3d"]["thread_num"].IsDefined() ? config["bnb3d"]["thread_num"].as<int>() : 4;
        match_option.filter_size_scan = config["bnb3d"]["filter_size_scan"].IsDefined() ? config["bnb3d"]["filter_size_scan"].as<double>() : 0.1;
        match_option.debug_mode = config["bnb3d"]["debug_mode"].IsDefined() ? config["bnb3d"]["debug_mode"].as<bool>() : false;

        Pose lidar_extrinsic;
        lidar_extrinsic.x = config["relocalization_cfg"]["lidar_ext/x"].IsDefined() ? config["relocalization_cfg"]["lidar_ext/x"].as<double>() : 0.;
        lidar_extrinsic.y = config["relocalization_cfg"]["lidar_ext/y"].IsDefined() ? config["relocalization_cfg"]["lidar_ext/y"].as<double>() : 0.;
        lidar_extrinsic.z = config["relocalization_cfg"]["lidar_ext/z"].IsDefined() ? config["relocalization_cfg"]["lidar_ext/z"].as<double>() : 0.;
        lidar_extrinsic.roll = config["relocalization_cfg"]["lidar_ext/roll"].IsDefined() ? config["relocalization_cfg"]["lidar_ext/roll"].as<double>() : 0.;
        lidar_extrinsic.pitch = config["relocalization_cfg"]["lidar_ext/pitch"].IsDefined() ? config["relocalization_cfg"]["lidar_ext/pitch"].as<double>() : 0.;
        lidar_extrinsic.yaw = config["relocalization_cfg"]["lidar_ext/yaw"].IsDefined() ? config["relocalization_cfg"]["lidar_ext/yaw"].as<double>() : 0.;
        slam.relocalization->set_bnb3d_param(match_option, lidar_extrinsic);

        double step_size, resolution;
        step_size = config["ndt"]["step_size"].IsDefined() ? config["ndt"]["step_size"].as<double>() : 0.1;
        resolution = config["ndt"]["resolution"].IsDefined() ? config["ndt"]["resolution"].as<double>() : 1;
        slam.relocalization->set_ndt_param(step_size, resolution);

        bool use_gicp;
        double gicp_downsample, filter_range, search_radius, teps, feps, fitness_score;
        use_gicp = config["gicp"]["use_gicp"].IsDefined() ? config["gicp"]["use_gicp"].as<bool>() : true;
        filter_range = config["gicp"]["filter_range"].IsDefined() ? config["gicp"]["filter_range"].as<double>() : 80;
        gicp_downsample = config["gicp"]["gicp_downsample"].IsDefined() ? config["gicp"]["gicp_downsample"].as<double>() : 0.2;
        search_radius = config["gicp"]["search_radius"].IsDefined() ? config["gicp"]["search_radius"].as<double>() : 0.5;
        teps = config["gicp"]["teps"].IsDefined() ? config["gicp"]["teps"].as<double>() : 1e-3;
        feps = config["gicp"]["feps"].IsDefined() ? config["gicp"]["feps"].as<double>() : 1e-3;
        fitness_score = config["gicp"]["fitness_score"].IsDefined() ? config["gicp"]["fitness_score"].as<double>() : 0.3;
        slam.relocalization->set_gicp_param(use_gicp, filter_range, gicp_downsample, search_radius, teps, feps, fitness_score);
    }

    auto frontend_type = config["mapping"]["frontend_type"].IsDefined() ? config["mapping"]["frontend_type"].as<int>() : 0;
    if (frontend_type == Fastlio)
    {
        slam.frontend = make_shared<FastlioOdometry>();
        LOG_WARN("frontend use fastlio!");
    }
    else if (frontend_type == Pointlio)
    {
        slam.frontend = make_shared<PointlioOdometry>();
        LOG_WARN("frontend use pointlio!");
        auto pointlio = dynamic_cast<PointlioOdometry *>(slam.frontend.get());
        pointlio->imu_en = config["mapping"]["imu_en"].IsDefined() ? config["mapping"]["imu_en"].as<bool>() : true;
        pointlio->use_imu_as_input = config["mapping"]["use_imu_as_input"].IsDefined() ? config["mapping"]["use_imu_as_input"].as<bool>() : true;
        pointlio->prop_at_freq_of_imu = config["mapping"]["prop_at_freq_of_imu"].IsDefined() ? config["mapping"]["prop_at_freq_of_imu"].as<bool>() : true;
        pointlio->check_saturation = config["mapping"]["check_saturation"].IsDefined() ? config["mapping"]["check_saturation"].as<bool>() : true;
        pointlio->saturation_acc = config["mapping"]["saturation_acc"].IsDefined() ? config["mapping"]["saturation_acc"].as<double>() : 3.0;
        pointlio->saturation_gyro = config["mapping"]["saturation_gyro"].IsDefined() ? config["mapping"]["saturation_gyro"].as<double>() : 35.0;
        double acc_cov_output = config["mapping"]["acc_cov_output"].IsDefined() ? config["mapping"]["acc_cov_output"].as<double>() : 500;
        double gyr_cov_output = config["mapping"]["gyr_cov_output"].IsDefined() ? config["mapping"]["gyr_cov_output"].as<double>() : 1000;
        double gyr_cov_input = config["mapping"]["gyr_cov_input"].IsDefined() ? config["mapping"]["gyr_cov_input"].as<double>() : 0.01;
        double acc_cov_input = config["mapping"]["acc_cov_input"].IsDefined() ? config["mapping"]["acc_cov_input"].as<double>() : 0.1;
        double vel_cov = config["mapping"]["vel_cov"].IsDefined() ? config["mapping"]["vel_cov"].as<double>() : 20;
        pointlio->Q_input = process_noise_cov_input(gyr_cov_input, acc_cov_input, b_gyr_cov, b_acc_cov);
        pointlio->Q_output = process_noise_cov_output(vel_cov, gyr_cov_output, acc_cov_output, b_gyr_cov, b_acc_cov);
        double imu_meas_acc_cov = config["mapping"]["imu_meas_acc_cov"].IsDefined() ? config["mapping"]["imu_meas_acc_cov"].as<double>() : 0.1;
        double imu_meas_omg_cov = config["mapping"]["imu_meas_omg_cov"].IsDefined() ? config["mapping"]["imu_meas_omg_cov"].as<double>() : 0.1;
        pointlio->R_imu << imu_meas_omg_cov, imu_meas_omg_cov, imu_meas_omg_cov, imu_meas_acc_cov, imu_meas_acc_cov, imu_meas_acc_cov;
    }
    else
    {
        LOG_ERROR("frontend odom type error!");
        exit(100);
    }

    slam.frontend->lidar->init(n_scans, scan_rate, time_unit, blind, detect_range);
    slam.frontend->imu->imu_rate = config["preprocess"]["imu_rate"].IsDefined() ? config["preprocess"]["imu_rate"].as<int>() : 200;
    slam.frontend->imu->set_imu_cov(process_noise_cov(gyr_cov, acc_cov, b_gyr_cov, b_acc_cov));

    slam.frontend->timedelay_lidar2imu = config["common"]["timedelay_lidar2imu"].IsDefined() ? config["common"]["timedelay_lidar2imu"].as<double>() : 0;
    slam.frontend->gravity_init = config["mapping"]["gravity_init"].IsDefined() ? config["mapping"]["gravity_init"].as<vector<double>>() : vector<double>();

    slam.frontend->num_max_iterations = config["mapping"]["max_iteration"].IsDefined() ? config["mapping"]["max_iteration"].as<int>() : 4;
    slam.frontend->surf_frame_ds_res = config["mapping"]["surf_frame_ds_res"].IsDefined() ? config["mapping"]["surf_frame_ds_res"].as<double>() : 0.5;
    slam.frontend->point_skip_num = config["mapping"]["point_skip_num"].IsDefined() ? config["mapping"]["point_skip_num"].as<int>() : 2;
    slam.frontend->space_down_sample = config["mapping"]["space_down_sample"].IsDefined() ? config["mapping"]["space_down_sample"].as<bool>() : true;
    slam.frontend->ikdtree_resolution = config["mapping"]["ikdtree_resolution"].IsDefined() ? config["mapping"]["ikdtree_resolution"].as<double>() : 0.5;
    slam.frontend->lidar_model_search_range = config["mapping"]["lidar_model_search_range"].IsDefined() ? config["mapping"]["lidar_model_search_range"].as<double>() : 5;
    slam.frontend->lidar_meas_cov = config["mapping"]["lidar_meas_cov"].IsDefined() ? config["mapping"]["lidar_meas_cov"].as<double>() : 0.001;
    slam.frontend->cube_len = config["mapping"]["cube_side_length"].IsDefined() ? config["mapping"]["cube_side_length"].as<double>() : 200;
    slam.frontend->extrinsic_est_en = config["mapping"]["extrinsic_est_en"].IsDefined() ? config["mapping"]["extrinsic_est_en"].as<bool>() : true;
    slam.frontend->loger.runtime_log = config["mapping"]["runtime_log_enable"].IsDefined() ? config["mapping"]["runtime_log_enable"].as<int>() : 0;

    extrinT = config["mapping"]["extrinsic_T"].IsDefined() ? config["mapping"]["extrinsic_T"].as<vector<double>>() : vector<double>();
    extrinR = config["mapping"]["extrinsic_R"].IsDefined() ? config["mapping"]["extrinsic_R"].as<vector<double>>() : vector<double>();
    extrinT_eigen << VEC_FROM_ARRAY(extrinT);
    extrinR_eigen << MAT_FROM_ARRAY(extrinR);
    slam.frontend->set_extrinsic(extrinT_eigen, extrinR_eigen);

    slam.init_system_mode();
}
