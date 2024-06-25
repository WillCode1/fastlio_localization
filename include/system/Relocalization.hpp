#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/gicp.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/impl/pcl_base.hpp>
#include <pcl/filters/impl/voxel_grid.hpp>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include "system/Header.h"
#include "GnssProcessor.hpp"
#include "global_localization/bnb3d.h"
#include "global_localization/scancontext/Scancontext.h"
// #define gnss_with_direction

class Relocalization
{
public:
    Relocalization();
    ~Relocalization();
    void set_extrinsic(const V3D &transl, const M3D &rot);
    bool load_prior_map(const PointCloudType::Ptr &global_map);
    bool load_keyframe_descriptor(const std::string &path);
    bool run(const PointCloudType::Ptr &scan, Eigen::Matrix4d &result, const double &lidar_beg_time);

    void set_init_pose(const Pose &_manual_pose);
    void set_bnb3d_param(const BnbOptions &match_option, const Pose &lidar_pose);
    void set_ndt_param(const double &_step_size, const double &_resolution);
    void set_gicp_param(bool _use_gicp, const double &filter_range, const double &gicp_ds, const double &search_radi, const double &tep, const double &fep, const double &fit_score);
    void add_keyframe_descriptor(const PointCloudType::Ptr thiskeyframe, const std::string &path);
    void get_pose_score(const Eigen::Matrix4d &imu_pose, const PointCloudType::Ptr &scan, double &bnb_score, double &ndt_score);

    std::string algorithm_type = "UNKNOW";
    BnbOptions bnb_option;
    Pose manual_pose, lidar_extrinsic, rough_pose;
    std::shared_ptr<BranchAndBoundMatcher3D> bnb3d;

    pcl::PointCloud<PointXYZIRPYT>::Ptr trajectory_poses;
    std::shared_ptr<ScanContext::SCManager> sc_manager; // scan context

    GnssPose gnss_pose;
    utm_coordinate::utm_point utm_origin;
    Eigen::Matrix4d extrinsic_imu2gnss;

private:
    bool fine_tune_pose(PointCloudType::Ptr scan, Eigen::Matrix4d &result, const Eigen::Matrix4d &lidar_ext, const double &score);
    bool run_gnss_relocalization(PointCloudType::Ptr scan, Eigen::Matrix4d &result, const double &lidar_beg_time, const Eigen::Matrix4d &lidar_ext, double &score);
    bool run_scan_context(PointCloudType::Ptr scan, Eigen::Matrix4d &rough_mat, const Eigen::Matrix4d &lidar_ext, double &score);
    bool run_manually_set(PointCloudType::Ptr scan, Eigen::Matrix4d &rough_mat, const Eigen::Matrix4d &lidar_ext, double &score);

    bool prior_pose_inited = false;

    // ndt
    double step_size = 0.1;
    double resolution = 1;

    // gicp
    bool use_gicp = true;
    double filter_range = 80;
    double gicp_downsample = 0.2;
    double search_radius = 0.2;
    double teps = 0.001;
    double feps = 0.001;
    double fitness_score = 0.3;

    pcl::VoxelGrid<PointType> voxel_filter;
    pcl::NormalDistributionsTransform<PointType, PointType> ndt;
    pcl::GeneralizedIterativeClosestPoint<PointType, PointType> gicp;
};

Relocalization::Relocalization()
{
    sc_manager = std::make_shared<ScanContext::SCManager>();
    trajectory_poses.reset(new pcl::PointCloud<PointXYZIRPYT>());
    extrinsic_imu2gnss.setIdentity();
}

Relocalization::~Relocalization()
{
}

void Relocalization::set_extrinsic(const V3D &transl, const M3D &rot)
{
  extrinsic_imu2gnss.setIdentity();
  extrinsic_imu2gnss.topLeftCorner(3, 3) = rot;
  extrinsic_imu2gnss.topRightCorner(3, 1) = transl;
}

bool Relocalization::run_gnss_relocalization(PointCloudType::Ptr scan, Eigen::Matrix4d &rough_mat, const double &lidar_beg_time, const Eigen::Matrix4d &lidar_ext, double &score)
{
    Timer timer;
    if (std::abs(lidar_beg_time - gnss_pose.timestamp) > 0.5)
    {
        LOG_WARN("gnss relocalization failed! time interval = %f.", lidar_beg_time - gnss_pose.timestamp);
        return false;
    }

    utm_coordinate::geographic_position lla;
    utm_coordinate::utm_point utm;
    lla.latitude = gnss_pose.gnss_position(0);
    lla.longitude = gnss_pose.gnss_position(1);
    lla.altitude = gnss_pose.gnss_position(2);
    utm_coordinate::LLAtoUTM(lla, utm);

    if (utm.zone.compare(utm_origin.zone) != 0)
    {
        LOG_ERROR("utm zone inconsistency!");
        return false;
    }

    gnss_pose.gnss_position = V3D(utm.east - utm_origin.east, utm.north - utm_origin.north, utm.up - utm_origin.up);
    rough_mat = Eigen::Matrix4d::Identity();
    rough_mat.topLeftCorner(3, 3) = gnss_pose.gnss_quat.toRotationMatrix();
    rough_mat.topRightCorner(3, 1) = gnss_pose.gnss_position;
    rough_mat *= extrinsic_imu2gnss;

    EigenMath::DecomposeAffineMatrix(rough_mat, rough_pose.x, rough_pose.y, rough_pose.z, rough_pose.roll, rough_pose.pitch, rough_pose.yaw);
    LOG_WARN("gnss relocalization success! pose = (%.2lf,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf)!",
             rough_pose.x, rough_pose.y, rough_pose.z, RAD2DEG(rough_pose.roll), RAD2DEG(rough_pose.pitch), RAD2DEG(rough_pose.yaw));

    bool bnb_success = true;
    auto bnb_opt_tmp = bnb_option;
    bnb_opt_tmp.min_score = 0.1;
    bnb_opt_tmp.linear_xy_window_size = 2;
    bnb_opt_tmp.linear_z_window_size = 0.5;
    bnb_opt_tmp.min_xy_resolution = 0.2;
    bnb_opt_tmp.min_z_resolution = 0.1;
#ifdef gnss_with_direction
    bnb_opt_tmp.angular_search_window = DEG2RAD(6);
    bnb_opt_tmp.min_angular_resolution = DEG2RAD(1);
#else
    bnb_opt_tmp.angular_search_window = DEG2RAD(180);
    bnb_opt_tmp.min_angular_resolution = DEG2RAD(5);
#endif
    if (!bnb3d->MatchWithMatchOptions(rough_pose, rough_pose, scan, bnb_opt_tmp, lidar_ext, score))
    {
        bnb_success = false;
        LOG_ERROR("bnb_failed, when bnb min_score = %.2f!", bnb_opt_tmp.min_score);
    }
    if (bnb_success)
    {
        LOG_INFO("bnb_success!");
        LOG_WARN("bnb_pose = (%.2lf,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf), score_cnt = %d, time = %.2lf ms",
                 rough_pose.x, rough_pose.y, rough_pose.z, RAD2DEG(rough_pose.roll), RAD2DEG(rough_pose.pitch), RAD2DEG(rough_pose.yaw),
                 bnb3d->sort_cnt, timer.elapsedLast());
    }
    return true;
}

bool Relocalization::run_scan_context(PointCloudType::Ptr scan, Eigen::Matrix4d &rough_mat, const Eigen::Matrix4d &lidar_ext, double &score)
{
    Timer timer;
    PointCloudType::Ptr scanDS(new PointCloudType());
    pcl::PointCloud<SCPointType>::Ptr sc_input(new pcl::PointCloud<SCPointType>());
    voxel_filter.setLeafSize(0.5, 0.5, 0.5);
    voxel_filter.setInputCloud(scan);
    voxel_filter.filter(*scanDS);

    pcl::PointXYZI tmp;
    for (auto &point : scanDS->points)
    {
        tmp.x = point.x;
        tmp.y = point.y;
        tmp.z = point.z;
        sc_input->push_back(tmp);
    }
    auto sc_res = sc_manager->relocalize(*sc_input);
    if (sc_res.first != -1 && sc_res.first < trajectory_poses->size())
    {
        const auto &pose_ref = trajectory_poses->points[sc_res.first];
        // lidar pose -> imu pose
        rough_mat = EigenMath::CreateAffineMatrix(V3D(pose_ref.x, pose_ref.y, pose_ref.z), V3D(pose_ref.roll, pose_ref.pitch, pose_ref.yaw + sc_res.second));
        rough_mat *= lidar_ext.inverse();
        EigenMath::DecomposeAffineMatrix(rough_mat, rough_pose.x, rough_pose.y, rough_pose.z, rough_pose.roll, rough_pose.pitch, rough_pose.yaw);
        LOG_WARN("scan context success! res index = %d, pose = (%.2lf,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf)!", sc_res.first,
                 rough_pose.x, rough_pose.y, rough_pose.z, RAD2DEG(rough_pose.roll), RAD2DEG(rough_pose.pitch), RAD2DEG(rough_pose.yaw));

        bool bnb_success = true;
        auto bnb_opt_tmp = bnb_option;
        bnb_opt_tmp.min_score = 0.1;
        bnb_opt_tmp.linear_xy_window_size = 2;
        bnb_opt_tmp.linear_z_window_size = 0.5;
        bnb_opt_tmp.min_xy_resolution = 0.2;
        bnb_opt_tmp.min_z_resolution = 0.1;
        bnb_opt_tmp.angular_search_window = DEG2RAD(6);
        bnb_opt_tmp.min_angular_resolution = DEG2RAD(1);
        if (!bnb3d->MatchWithMatchOptions(rough_pose, rough_pose, scan, bnb_opt_tmp, lidar_ext, score))
        {
            bnb_success = false;
            LOG_ERROR("bnb_failed, when bnb min_score = %.2f!", bnb_opt_tmp.min_score);
        }
        if (bnb_success)
        {
            LOG_INFO("bnb_success!");
            LOG_WARN("bnb_pose = (%.2lf,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf), score_cnt = %d, time = %.2lf ms",
                     rough_pose.x, rough_pose.y, rough_pose.z, RAD2DEG(rough_pose.roll), RAD2DEG(rough_pose.pitch), RAD2DEG(rough_pose.yaw),
                     bnb3d->sort_cnt, timer.elapsedLast());
        }
    }
    else
    {
        LOG_ERROR("scan context failed, res index = %d, total descriptors = %lu! Please move the vehicle to another position and try again.", sc_res.first, trajectory_poses->size());
        return false;
    }
    return true;
}

bool Relocalization::run_manually_set(PointCloudType::Ptr scan, Eigen::Matrix4d &rough_mat, const Eigen::Matrix4d &lidar_ext, double &score)
{
    if (!prior_pose_inited)
    {
        LOG_WARN("wait for the boot position to be manually set!");
        return false;
    }

    Timer timer;
    bool bnb_success = true;
    if (!bnb3d->MatchWithMatchOptions(manual_pose, rough_pose, scan, bnb_option, lidar_ext, score))
    {
        auto bnb_opt_tmp = bnb_option;
        bnb_opt_tmp.min_score = 0.1;
        LOG_ERROR("bnb_failed, when bnb min_score = %.2f! min_score set to %.2f and try again.", bnb_option.min_score, bnb_opt_tmp.min_score);
        if (!bnb3d->MatchWithMatchOptions(manual_pose, rough_pose, scan, bnb_opt_tmp, lidar_ext, score))
        {
            bnb_success = false;
            rough_pose = manual_pose;
            LOG_ERROR("bnb_failed, when bnb min_score = %.2f!", bnb_opt_tmp.min_score);
        }
    }
    if (bnb_success)
    {
        LOG_INFO("bnb_success!");
        LOG_WARN("bnb_pose = (%.2lf,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf), score_cnt = %d, time = %.2lf ms",
                 rough_pose.x, rough_pose.y, rough_pose.z, RAD2DEG(rough_pose.roll), RAD2DEG(rough_pose.pitch), RAD2DEG(rough_pose.yaw),
                 bnb3d->sort_cnt, timer.elapsedLast());
    }
    return true;
}

bool Relocalization::run(const PointCloudType::Ptr &scan, Eigen::Matrix4d &result, const double &lidar_beg_time)
{
    Eigen::Matrix4d lidar_ext = lidar_extrinsic.toMatrix4d();
    bool success_flag = true;
    double score = 0;

    if (run_gnss_relocalization(scan, result, lidar_beg_time, lidar_ext, score) && fine_tune_pose(scan, result, lidar_ext, score))
    {
        LOG_WARN("relocalization successfully!!!!!!");
        return true;
    }

    if (algorithm_type.compare("scan_context") == 0)
    {
        if (!run_scan_context(scan, result, lidar_ext, score) || !fine_tune_pose(scan, result, lidar_ext, score))
        {
#ifdef DEDUB_MODE
            result = EigenMath::CreateAffineMatrix(V3D(rough_pose.x, rough_pose.y, rough_pose.z), V3D(rough_pose.roll, rough_pose.pitch, rough_pose.yaw));
#endif
            success_flag = false;
        }
    }
    if (algorithm_type.compare("manually_set") == 0 || !success_flag)
    {
        success_flag = true;
        if (!run_manually_set(scan, result, lidar_ext, score) || !fine_tune_pose(scan, result, lidar_ext, score))
        {
#ifdef DEDUB_MODE
            result = EigenMath::CreateAffineMatrix(V3D(manual_pose.x, manual_pose.y, manual_pose.z), V3D(manual_pose.roll, manual_pose.pitch, manual_pose.yaw));
#endif
            success_flag = false;
        }
    }

    if (!success_flag)
    {
        LOG_ERROR("relocalization failed!");
        return false;
    }

    LOG_WARN("relocalization successfully!!!!!!");
    return true;
}

bool Relocalization::load_keyframe_descriptor(const std::string &path)
{
    if (!fs::exists(path))
        return false;

    int scd_file_count = 0, num_digits = 0;
    scd_file_count = FileOperation::getFilesNumByExtension(path, ".scd");

    if (scd_file_count != trajectory_poses->size())
        return false;

    num_digits = FileOperation::getOneFilenameByExtension(path, ".scd").length() - std::string(".scd").length();

    sc_manager->loadPriorSCD(path, num_digits, trajectory_poses->size());
    return true;
}

bool Relocalization::load_prior_map(const PointCloudType::Ptr &global_map)
{
    bnb3d = std::make_shared<BranchAndBoundMatcher3D>(global_map, bnb_option);
    ndt.setInputTarget(global_map);
    ndt.setMaximumIterations(150);
    ndt.setTransformationEpsilon(teps);
    ndt.setStepSize(step_size);
    ndt.setResolution(resolution);

    if (use_gicp)
    {
        gicp.setInputTarget(global_map);
        gicp.setMaximumIterations(150);
        gicp.setMaxCorrespondenceDistance(search_radius);
        gicp.setTransformationEpsilon(teps);
        gicp.setEuclideanFitnessEpsilon(feps);
    }
    return true;
}

bool Relocalization::fine_tune_pose(PointCloudType::Ptr scan, Eigen::Matrix4d &result, const Eigen::Matrix4d &lidar_ext, const double &score)
{
    Timer timer;
    result = EigenMath::CreateAffineMatrix(V3D(rough_pose.x, rough_pose.y, rough_pose.z), V3D(rough_pose.roll, rough_pose.pitch, rough_pose.yaw));
    if (score >= bnb_option.enough_score)
    {
        LOG_WARN("bnb score greater than %.2f, enough!", bnb_option.enough_score);
        return true;
    }

    result *= lidar_ext; // imu pose -> lidar pose

    PointCloudType::Ptr filter(new PointCloudType());
    for (auto &point : scan->points)
        if (pointDistanceSquare(point) < filter_range * filter_range)
            filter->push_back(point);

    voxel_filter.setLeafSize(gicp_downsample, gicp_downsample, gicp_downsample);
    voxel_filter.setInputCloud(filter);
    voxel_filter.filter(*filter);

    PointCloudType::Ptr aligned(new PointCloudType());
    ndt.setInputSource(filter);
    ndt.align(*aligned, result.cast<float>());

    if (!ndt.hasConverged())
    {
        LOG_ERROR("NDT not converge!");
        return false;
    }
    else if (ndt.getFitnessScore() > fitness_score)
    {
        LOG_ERROR("failed! NDT fitness_score = %f.", ndt.getFitnessScore());
        return false;
    }
    if (ndt.getFitnessScore() < 0.1)
    {
        LOG_WARN("NDT fitness_score = %f.", ndt.getFitnessScore());
    }
    else
    {
        LOG_ERROR("NDT fitness_score = %f.", ndt.getFitnessScore());
    }

    result = ndt.getFinalTransformation().cast<double>();
    result *= lidar_ext.inverse();

    Eigen::Vector3d pos, euler;
    EigenMath::DecomposeAffineMatrix(result, pos, euler);
    LOG_WARN("ndt pose = (%.2lf,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf), ndt_time = %.2lf ms, ndt_iters = %d",
             pos(0), pos(1), pos(2), RAD2DEG(euler(0)), RAD2DEG(euler(1)), RAD2DEG(euler(2)), timer.elapsedLast(), ndt.getFinalNumIteration());

    if (use_gicp)
    {
        gicp.setInputSource(filter);
        gicp.align(*aligned, ndt.getFinalTransformation());

        if (!gicp.hasConverged())
        {
            LOG_ERROR("GICP not converge!");
            return false;
        }
        else if (gicp.getFitnessScore() > fitness_score)
        {
            LOG_ERROR("failed! GICP fitness_score = %f.", gicp.getFitnessScore());
            return false;
        }
        if (gicp.getFitnessScore() < 0.1)
        {
            LOG_WARN("GICP fitness_score = %f.", gicp.getFitnessScore());
        }
        else
        {
            LOG_ERROR("GICP fitness_score = %f.", gicp.getFitnessScore());
        }

        result = gicp.getFinalTransformation().cast<double>();
        result *= lidar_ext.inverse();

        EigenMath::DecomposeAffineMatrix(result, pos, euler);
        LOG_WARN("gicp pose = (%.2lf,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf), gicp_time = %.2lf ms",
                 pos(0), pos(1), pos(2), RAD2DEG(euler(0)), RAD2DEG(euler(1)), RAD2DEG(euler(2)), timer.elapsedLast());
    }
    return true;
}

void Relocalization::set_init_pose(const Pose &_manual_pose)
{
    manual_pose = _manual_pose; // imu_pose
    manual_pose.roll -= lidar_extrinsic.roll;
    manual_pose.pitch -= lidar_extrinsic.pitch;
    manual_pose.yaw -= lidar_extrinsic.yaw;

    {
        std::vector<int> indices;
        std::vector<float> distances;
        pcl::KdTreeFLANN<PointXYZIRPYT> kdtree;
        PointXYZIRPYT manual_pos;
        manual_pos.x = manual_pose.x;
        manual_pos.y = manual_pose.y;
        manual_pos.z = manual_pose.z;
        kdtree.setInputCloud(trajectory_poses);
        kdtree.radiusSearch(manual_pos, 20, indices, distances, 1);

        if (indices.size() == 1)
        {
            auto pose_ref = trajectory_poses->points[indices.back()];
            Eigen::Matrix4d rough_mat = EigenMath::CreateAffineMatrix(V3D(pose_ref.x, pose_ref.y, pose_ref.z), V3D(pose_ref.roll, pose_ref.pitch, pose_ref.yaw));
            Eigen::Matrix4d lidar_ext = lidar_extrinsic.toMatrix4d();
            rough_mat *= lidar_ext.inverse();
            double tmp;
            EigenMath::DecomposeAffineMatrix(rough_mat, tmp, tmp, manual_pose.z, tmp, tmp, tmp);
            bnb_option.linear_z_window_size = 1;
        }
        else
        {
            bnb_option.linear_z_window_size = 6;
            LOG_ERROR("manual position no elevation found!");
        }
    }

    LOG_WARN("*******************************************");
    LOG_WARN("set_init_pose = (%.2lf,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf)", manual_pose.x, manual_pose.y, manual_pose.z,
             RAD2DEG(manual_pose.roll), RAD2DEG(manual_pose.pitch), RAD2DEG(manual_pose.yaw));
    LOG_WARN("*******************************************");
    prior_pose_inited = true;
}

void Relocalization::set_bnb3d_param(const BnbOptions &match_option, const Pose &lidar_pose)
{
    bnb_option = match_option;
    LOG_WARN("*********** BnB Localizer Param ***********");
    LOG_WARN("linear_xy_window_size: %lf m", bnb_option.linear_xy_window_size);
    LOG_WARN("linear_z_window_size: %lf m", bnb_option.linear_z_window_size);
    LOG_WARN("angular_search_window: %lf degree", bnb_option.angular_search_window);
    std::stringstream resolutions;
    for (const auto &resolution : bnb_option.pc_resolutions)
    {
        resolutions << resolution << " ";
    }
    LOG_WARN("pc_resolutions: [ %s]", resolutions.str().c_str());
    LOG_WARN("bnb_depth: %d", bnb_option.bnb_depth);
    LOG_WARN("min_score: %lf", bnb_option.min_score);
    LOG_WARN("enough_score: %lf", bnb_option.enough_score);
    LOG_WARN("min_xy_resolution: %lf", bnb_option.min_xy_resolution);
    LOG_WARN("min_z_resolution: %lf", bnb_option.min_z_resolution);
    LOG_WARN("min_angular_resolution: %lf", bnb_option.min_angular_resolution);
    LOG_WARN("filter_size_scan: %lf", bnb_option.filter_size_scan);
    LOG_WARN("debug_mode: %d", bnb_option.debug_mode);
    LOG_WARN("*******************************************");

    bnb_option.angular_search_window = DEG2RAD(bnb_option.angular_search_window);
    bnb_option.min_angular_resolution = DEG2RAD(bnb_option.min_angular_resolution);

    lidar_extrinsic = lidar_pose;
    LOG_WARN("*******************************************");
    LOG_WARN("lidar_ext = (%.2lf,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf)", lidar_extrinsic.x, lidar_extrinsic.y, lidar_extrinsic.z,
             lidar_extrinsic.roll, lidar_extrinsic.pitch, lidar_extrinsic.yaw);
    LOG_WARN("*******************************************");
}

void Relocalization::set_ndt_param(const double &_step_size, const double &_resolution)
{
    step_size = _step_size;
    resolution = _resolution;
}

void Relocalization::set_gicp_param(bool _use_gicp, const double &_filter_range, const double &gicp_ds, const double &search_radi, const double &tep, const double &fep, const double &fit_score)
{
    use_gicp = _use_gicp;
    filter_range = _filter_range;
    gicp_downsample = gicp_ds;
    search_radius = search_radi;
    teps = tep;
    feps = fep;
    fitness_score = fit_score;
}

void Relocalization::add_keyframe_descriptor(const PointCloudType::Ptr thiskeyframe, const std::string &path)
{
    PointCloudType::Ptr thiskeyframeDS(new PointCloudType());
    pcl::PointCloud<SCPointType>::Ptr sc_input(new pcl::PointCloud<SCPointType>());
    voxel_filter.setLeafSize(0.5, 0.5, 0.5);
    voxel_filter.setInputCloud(thiskeyframe);
    voxel_filter.filter(*thiskeyframeDS);

    pcl::PointXYZI tmp;
    for (auto &point : thiskeyframeDS->points)
    {
        tmp.x = point.x;
        tmp.y = point.y;
        tmp.z = point.z;
        sc_input->push_back(tmp);
    }
    sc_manager->makeAndSaveScancontextAndKeys(*sc_input);

    if (path.compare("") != 0)
        sc_manager->saveCurrentSCD(path);
}

void Relocalization::get_pose_score(const Eigen::Matrix4d &imu_pose, const PointCloudType::Ptr &scan, double &bnb_score, double &ndt_score)
{
    PointCloudType::Ptr trans_pc(new PointCloudType(scan->points.size(), 1));
    const Eigen::Matrix4d &lidar_pose = imu_pose * lidar_extrinsic.toMatrix4d();
    pcl::transformPointCloud(*scan, *trans_pc, lidar_pose);
    bnb_score = bnb3d->calculateOccupancyScore(0, trans_pc);

    // getFitnessScore
    ndt.setInputSource(trans_pc);
    if (!ndt.initCompute())
        return;

    std::vector<int> nn_indices(1);
    std::vector<float> nn_dists(1);

    // For each point in the source dataset
    int nr = 0;
    for (std::size_t i = 0; i < trans_pc->points.size(); ++i)
    {
        // Find its nearest neighbor in the target
        ndt.getSearchMethodTarget()->nearestKSearch(trans_pc->points[i], 1, nn_indices, nn_dists);

        // Deal with occlusions (incomplete targets)
        if (nn_dists[0] <= std::numeric_limits<double>::max())
        {
            // Add to the fitness score
            ndt_score += nn_dists[0];
            nr++;
        }
    }

    if (nr > 0)
        ndt_score = ndt_score / nr;
    else
        ndt_score = std::numeric_limits<double>::max();
}
