#include <csignal>
#include <unistd.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>

#include "system/Header.h"
#include "system/ParametersRos2.h"
#include "system/System.hpp"
#include "livox_ros_driver2/msg/custom_msg.hpp"

#define MEASURES_BUFFER
// #define WORK
#ifdef WORK
#include "localization_msg/vehicle_pose.h"
// #include "robot_msgs/Level.h"
// #include "robot_msgs/ModuleStatus.h"
#endif

int lidar_type;
System slam;
std::string map_frame;
std::string lidar_frame;
std::string baselink_frame;
FILE *location_log = nullptr;
FILE *last_pose_record = nullptr;

bool path_en = true, scan_pub_en = false, dense_pub_en = false, lidar_tf_broadcast = false, imu_tf_broadcast = false;
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubLidarOdom;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubImuOdom;
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubImuPath;
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubrelocalizationDebug;
#ifdef WORK
rclcpp::Publisher<localization_msg::msg::vehicle_pose>::SharedPtr pubOdomDev;
rclcpp::Publisher<robot_msgs::msg::ModuleStatus>::SharedPtr pubModulesStatus;
#endif
std::unique_ptr<tf2_ros::Buffer> tf_buffer;
std::shared_ptr<tf2_ros::TransformListener> tf_listener;
std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

bool flg_exit = false;
void SigHandle(int sig)
{
    flg_exit = true;
    LOG_WARN("catch sig %d", sig);
}

void publish_cloud(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubCloud, PointCloudType::Ptr cloud, const double& lidar_end_time, const std::string& frame_id)
{
    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud, cloud_msg);
    cloud_msg.header.stamp = rclcpp::Time(lidar_end_time * 1e9);
    cloud_msg.header.frame_id = frame_id;
    pubCloud->publish(cloud_msg);
}

void publish_cloud_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubLaserCloudFull, PointCloudType::Ptr laserCloud, const state_ikfom &state, const double& lidar_end_time)
{
    PointCloudType::Ptr laserCloudWorld(new PointCloudType(laserCloud->size(), 1));
    pointcloudLidarToWorld(laserCloud, laserCloudWorld, state);
    publish_cloud(pubLaserCloudFull, laserCloudWorld, lidar_end_time, map_frame);
}

template <typename T>
void set_posestamp(T &out, const QD &rot, const V3D &pos)
{
    out.pose.position.x = pos.x();
    out.pose.position.y = pos.y();
    out.pose.position.z = pos.z();
    out.pose.orientation.x = rot.x();
    out.pose.orientation.y = rot.y();
    out.pose.orientation.z = rot.z();
    out.pose.orientation.w = rot.w();
}

void publish_tf(const QD &rot, const V3D &pos, const double &lidar_end_time)
{
    geometry_msgs::msg::TransformStamped transform_stamped;
    transform_stamped.header.stamp = rclcpp::Time(lidar_end_time * 1e9);
    transform_stamped.header.frame_id = map_frame;
    transform_stamped.child_frame_id = baselink_frame;
    transform_stamped.transform.translation.x = pos.x();
    transform_stamped.transform.translation.y = pos.y();
    transform_stamped.transform.translation.z = pos.z();
    transform_stamped.transform.rotation.x = rot.x();
    transform_stamped.transform.rotation.y = rot.y();
    transform_stamped.transform.rotation.z = rot.z();
    transform_stamped.transform.rotation.w = rot.w();
    tf_broadcaster->sendTransform(transform_stamped);
}

void publish_odometry(rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr &pubLidarOdom, const state_ikfom &state, const double &lidar_end_time, QD &baselink_rot, V3D &baselink_pos)
{
    static bool first_time = true;
    static Eigen::Quaterniond lidar2baselink_q;
    static Eigen::Vector3d lidar2baselink_t;
    if (first_time)
    {
        first_time = false;
        geometry_msgs::msg::TransformStamped transformStamped;
        try
        {
            transformStamped = tf_buffer->lookupTransform(lidar_frame, baselink_frame, tf2::TimePointZero);
        }
        catch (tf2::TransformException &ex)
        {
            LOG_ERROR("%s", ex.what());
        }

        auto pos = transformStamped.transform.translation;
        auto quat = transformStamped.transform.rotation;
        lidar2baselink_q = Eigen::Quaterniond(quat.w, quat.x, quat.y, quat.z);
        lidar2baselink_t = Eigen::Vector3d(pos.x, pos.y, pos.z);
    }

    // extrinsic lidar to baselink
    Eigen::Matrix4d lidar2baselink = Eigen::Matrix4d::Identity();
    lidar2baselink.topLeftCorner(3, 3) = lidar2baselink_q.toRotationMatrix();
    lidar2baselink.topRightCorner(3, 1) = lidar2baselink_t;

    // extrinsic lidar to imu
    Eigen::Matrix4d lidar2imu = Eigen::Matrix4d::Identity();
    lidar2imu.topLeftCorner(3, 3) = state.offset_R_L_I.toRotationMatrix();
    lidar2imu.topRightCorner(3, 1) = state.offset_T_L_I;

    // imu pose
    Eigen::Matrix4d pose_mat = Eigen::Matrix4d::Identity();
    pose_mat.topLeftCorner(3, 3) = state.rot.toRotationMatrix();
    pose_mat.topRightCorner(3, 1) = state.pos;

    // baselink pose
    auto baselink2imu = lidar2baselink * lidar2imu.inverse();
    pose_mat = pose_mat * baselink2imu.inverse();
    baselink_rot = QD(M3D(pose_mat.topLeftCorner(3, 3)));
    baselink_pos = pose_mat.topRightCorner(3, 1);

    if (std::abs(baselink_pos.x()) > 1e7 || std::abs(baselink_pos.y()) > 1e7 || std::abs(baselink_pos.z()) > 1e7)
    {
        LOG_WARN("localization state maybe abnormal! (baselink frame) pos(%f, %f, %f)", baselink_pos.x(), baselink_pos.y(), baselink_pos.z());
    }

    nav_msgs::msg::Odometry odomAftMapped;
    odomAftMapped.header.frame_id = map_frame;
    odomAftMapped.child_frame_id = baselink_frame;
    odomAftMapped.header.stamp = rclcpp::Time(lidar_end_time * 1e9);
    set_posestamp(odomAftMapped.pose, baselink_rot, baselink_pos);
    pubLidarOdom->publish(odomAftMapped);
    if (lidar_tf_broadcast)
        publish_tf(baselink_rot, baselink_pos, lidar_end_time);
}

#ifdef WORK
void publish_module_status(const double &time, int level)
{
    // robot_msgs::ModuleStatus status;
    // status.header.stamp = ros::Time().fromSec(time);
    // status.header.frame_id = "LOCATION";
    // status.level = level;
    // robot_msgs::ModuleStatusItem item;
    // item.error_code = 4000101;
    // item.level = level;
    // status.items.emplace_back(item);
    // pubModulesStatus.publish(status);
}

void publish_odometry2(rclcpp::Publisher<localization_msg::msg::vehicle_pose>::SharedPtr &pubOdomDev, const state_ikfom &state, const double &lidar_end_time, bool vaild, QD &baselink_rot, V3D &baselink_pos)
{
    localization_msg::vehicle_pose odom;
    odom.header.frame_id = map_frame;
    odom.header.stamp = rclcpp::Time(lidar_end_time * 1e9);
    odom.is_valid = vaild;

    static bool first_time = true;
    static Eigen::Quaterniond lidar2baselink_q;
    static Eigen::Vector3d lidar2baselink_t;
    if (first_time)
    {
        first_time = false;
        geometry_msgs::msg::TransformStamped transformStamped;
        try
        {
            transformStamped = tf_buffer->lookupTransform(lidar_frame, baselink_frame, tf2::TimePointZero);
        }
        catch (tf2::TransformException &ex)
        {
            LOG_ERROR("%s", ex.what());
        }

        auto pos = transformStamped.transform.translation;
        auto quat = transformStamped.transform.rotation;
        lidar2baselink_q = Eigen::Quaterniond(quat.w, quat.x, quat.y, quat.z);
        lidar2baselink_t = Eigen::Vector3d(pos.x, pos.y, pos.z);
    }

    // extrinsic lidar to baselink
    Eigen::Matrix4d lidar2baselink = Eigen::Matrix4d::Identity();
    lidar2baselink.topLeftCorner(3, 3) = lidar2baselink_q.toRotationMatrix();
    lidar2baselink.topRightCorner(3, 1) = lidar2baselink_t;

    // extrinsic lidar to imu
    Eigen::Matrix4d lidar2imu = Eigen::Matrix4d::Identity();
    lidar2imu.topLeftCorner(3, 3) = state.offset_R_L_I.toRotationMatrix();
    lidar2imu.topRightCorner(3, 1) = state.offset_T_L_I;

    // imu pose
    Eigen::Matrix4d pose_mat = Eigen::Matrix4d::Identity();
    pose_mat.topLeftCorner(3, 3) = state.rot.toRotationMatrix();
    pose_mat.topRightCorner(3, 1) = state.pos;

    // baselink pose
    auto baselink2imu = lidar2baselink * lidar2imu.inverse();
    pose_mat = pose_mat * baselink2imu.inverse();
    baselink_rot = QD(M3D(pose_mat.topLeftCorner(3, 3)));
    baselink_pos = pose_mat.topRightCorner(3, 1);

    odom.east = baselink_pos.x();
    odom.north = baselink_pos.y();
    odom.up = baselink_pos.z();
    auto res = EigenMath::Quaternion2RPY(baselink_rot);
    odom.roll = res(0);
    odom.pitch = res(1);
    odom.yaw = res(2);

    odom.east_vel = state.vel(0);
    odom.north_vel = state.vel(1);
    odom.up_vel = state.vel(2);
    odom.angular_vel[0] = slam.frontend->angular_velocity(0);
    odom.angular_vel[1] = slam.frontend->angular_velocity(1);
    odom.angular_vel[2] = slam.frontend->angular_velocity(2);
    odom.body_accel[0] = slam.frontend->linear_acceleration(0);
    odom.body_accel[1] = slam.frontend->linear_acceleration(1);
    odom.body_accel[2] = slam.frontend->linear_acceleration(2);

    if (std::abs(odom.east) > 1e7 || std::abs(odom.north) > 1e7 || std::abs(odom.up) > 1e7 ||
        std::abs(odom.east_vel) > 20 || std::abs(odom.north_vel) > 20 || std::abs(odom.up_vel) > 20 ||
        std::abs(odom.angular_vel[0]) > 100 || std::abs(odom.angular_vel[1]) > 100 || std::abs(odom.angular_vel[2]) > 100 ||
        std::abs(odom.body_accel[0]) > 100 || std::abs(odom.body_accel[1]) > 100 || std::abs(odom.body_accel[2]) > 100)
    {
        LOG_WARN("localization state maybe abnormal! (imu frame) pos(%f, %f, %f), vel(%f, %f, %f), ang_vel(%f, %f, %f), linear_acc(%f, %f, %f)",
                 odom.east, odom.north, odom.up, odom.east_vel, odom.north_vel, odom.up_vel,
                 odom.angular_vel[0], odom.angular_vel[1], odom.angular_vel[2], odom.body_accel[0], odom.body_accel[1], odom.body_accel[2]);
        odom.is_valid = false;
    }

    pubOdomDev.publish(odom);
    if (lidar_tf_broadcast)
        publish_tf(baselink_rot, baselink_pos, lidar_end_time);
}
#endif

void publish_imu_odometry(rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr &pubImuOdom, const state_ikfom &state, const double &imu_time)
{
    static bool first_time = true;
    static Eigen::Quaterniond lidar2baselink_q;
    static Eigen::Vector3d lidar2baselink_t;
    if (first_time)
    {
        first_time = false;
        geometry_msgs::msg::TransformStamped transformStamped;
        try
        {
            transformStamped = tf_buffer->lookupTransform(lidar_frame, baselink_frame, tf2::TimePointZero);
        }
        catch (tf2::TransformException &ex)
        {
            LOG_ERROR("%s", ex.what());
        }

        auto pos = transformStamped.transform.translation;
        auto quat = transformStamped.transform.rotation;
        lidar2baselink_q = Eigen::Quaterniond(quat.w, quat.x, quat.y, quat.z);
        lidar2baselink_t = Eigen::Vector3d(pos.x, pos.y, pos.z);
    }

    // extrinsic lidar to baselink
    Eigen::Matrix4d lidar2baselink = Eigen::Matrix4d::Identity();
    lidar2baselink.topLeftCorner(3, 3) = lidar2baselink_q.toRotationMatrix();
    lidar2baselink.topRightCorner(3, 1) = lidar2baselink_t;

    // extrinsic lidar to imu
    Eigen::Matrix4d lidar2imu = Eigen::Matrix4d::Identity();
    lidar2imu.topLeftCorner(3, 3) = state.offset_R_L_I.toRotationMatrix();
    lidar2imu.topRightCorner(3, 1) = state.offset_T_L_I;

    // imu pose
    Eigen::Matrix4d pose_mat = Eigen::Matrix4d::Identity();
    pose_mat.topLeftCorner(3, 3) = state.rot.toRotationMatrix();
    pose_mat.topRightCorner(3, 1) = state.pos;

    // baselink pose
    auto baselink2imu = lidar2baselink * lidar2imu.inverse();
    pose_mat = pose_mat * baselink2imu.inverse();
    QD baselink_rot = QD(M3D(pose_mat.topLeftCorner(3, 3)));
    V3D baselink_pos = pose_mat.topRightCorner(3, 1);

    if (std::abs(baselink_pos.x()) > 1e7 || std::abs(baselink_pos.y()) > 1e7 || std::abs(baselink_pos.z()) > 1e7)
    {
        LOG_WARN("localization state maybe abnormal! (baselink frame) pos(%f, %f, %f)", baselink_pos.x(), baselink_pos.y(), baselink_pos.z());
    }

    nav_msgs::msg::Odometry imuOdom;
    imuOdom.header.frame_id = map_frame;
    imuOdom.child_frame_id = baselink_frame;
    imuOdom.header.stamp = rclcpp::Time(imu_time * 1e9);
    set_posestamp(imuOdom.pose, baselink_rot, baselink_pos);
    pubImuOdom->publish(imuOdom);
    if (imu_tf_broadcast)
        publish_tf(baselink_rot, baselink_pos, imu_time);
}

void publish_imu_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr &pubPath, const QD &rot, const V3D &pos, const double &lidar_end_time)
{
    static geometry_msgs::msg::PoseStamped msg_body_pose;
    set_posestamp(msg_body_pose, rot, pos);
    msg_body_pose.header.stamp = rclcpp::Time(lidar_end_time * 1e9);
    msg_body_pose.header.frame_id = map_frame;

    static nav_msgs::msg::Path path;
    path.header.stamp = rclcpp::Clock().now();
    path.header.frame_id = map_frame;

    path.poses.push_back(msg_body_pose);
    pubPath->publish(path);
    /*** if path is too large, the rvis will crash ***/
    if (path.poses.size() >= 300)
    {
        path.poses.erase(path.poses.begin());
    }
}

#ifdef gnss_with_direction
void gnss_cbk(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    slam.relocalization->gnss_pose = GnssPose(msg->header.stamp.sec + msg->header.stamp.nanosec * 1.0e-9,
                                              V3D(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z),
                                              QD(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z));
}
#else
void gnss_cbk(const sensor_msgs::msg::NavSatFix::SharedPtr msg)
{
    slam.relocalization->gnss_pose = GnssPose(msg->header.stamp.sec + msg->header.stamp.nanosec * 1.0e-9, V3D(msg->latitude, msg->longitude, msg->altitude));
}
#endif

bool load_last_pose(const PointCloudType::Ptr &scan)
{
    static bool use_last_pose = true;
    if (!use_last_pose)
        return false;
    use_last_pose = false;
    LOG_WARN("relocate try to use last pose!");

    fseek(last_pose_record, 0, SEEK_SET);
    Eigen::Quaterniond quat;
    Eigen::Vector3d pos;
    auto res = fscanf(last_pose_record, "last_pose(xyz,rpy):( %lf %lf %lf %lf %lf %lf %lf )end!\n",
                      &pos.x(), &pos.y(), &pos.z(), &quat.w(), &quat.x(), &quat.y(), &quat.z());
    Eigen::Matrix4d imu_pose = Eigen::Matrix4d::Identity();
    imu_pose.topLeftCorner(3, 3) = quat.normalized().toRotationMatrix();
    imu_pose.topRightCorner(3, 1) = pos;
    slam.frontend->reset_state(imu_pose);

    if (res == -1)
    {
        LOG_ERROR("can't find last pose! parse error!");
        return false;
    }
    double bnb_score = 0, ndt_score = 0;
    Timer timer;
    slam.relocalization->get_pose_score(imu_pose, scan, bnb_score, ndt_score);
    if (bnb_score < 0.4)
    {
        LOG_ERROR("bnb_score too small = %.3f, should be greater than = %.3f!", bnb_score, 0.8);
        return false;
    }
    else if (ndt_score > 0.2)
    {
        LOG_ERROR("ndt_score too high = %.3f, should be less than = %.3f!", ndt_score, 0.1);
        return false;
    }
    else
    {
        LOG_WARN("relocate use last pose successfully! bnb_score = %.3f, ndt_score = %.3f. cost time = %.2lf ms.", bnb_score, ndt_score, timer.elapsedLast());
        slam.system_state_vaild = true;
    }

    return slam.system_state_vaild;
}

void sensor_data_process()
{
    if (flg_exit)
        return;

    if (!slam.frontend->sync_sensor_data())
        return;

    LOG_DEBUG("sync_sensor_data");
#ifdef MEASURES_BUFFER
    // for relocalization data cache
    static std::deque<shared_ptr<MeasureCollection>> measures_cache;
#endif

    if (!slam.system_state_vaild)
    {
        if (last_pose_record && load_last_pose(slam.frontend->measures->lidar))
        {
            // only for restart!
        }
        else if (!slam.run_relocalization_thread)
        {
            if (slam.relocalization_thread.joinable())
            {
                slam.relocalization_thread.join();
                measures_cache.clear();
                LOG_WARN("measures cache clear!");
            }

            PointCloudType::Ptr cur_scan(new PointCloudType);
            *cur_scan = *slam.frontend->measures->lidar;
            slam.relocalization_thread = std::thread(&System::run_relocalization, &slam, cur_scan, slam.frontend->measures->lidar_beg_time);
        }
        // publish_module_status(slam.frontend->measures->lidar_end_time, robot_msgs::Level::WARN);
#ifdef MEASURES_BUFFER
        measures_cache.emplace_back(std::make_shared<MeasureCollection>());
        *measures_cache.back() = *slam.frontend->measures;
#endif
        return;
    }

#ifdef MEASURES_BUFFER
    if (!measures_cache.empty())
    {
        measures_cache.emplace_back(std::make_shared<MeasureCollection>());
        *measures_cache.back() = *slam.frontend->measures;

        while (!measures_cache.empty())
        {
            slam.frontend->measures = measures_cache.front();

            QD baselink_rot;
            V3D baselink_pos;
            if (slam.run())
            {
                const auto &state = slam.frontend->get_state();
                LOG_INFO("location valid. feats_down = %d, cost time = %.1fms.", slam.frontend->loger.feats_down_size, slam.frontend->loger.total_time);
                slam.frontend->loger.print_pose(state, "cur_imu_pose");

                /******* Publish odometry *******/
#ifndef WORK
                publish_odometry(pubLidarOdom, state, slam.frontend->measures->lidar_end_time, baselink_rot, baselink_pos);
#else
                publish_odometry2(pubOdomDev, state, slam.frontend->measures->lidar_end_time, slam.system_state_vaild, baselink_rot, baselink_pos);
                // publish_module_status(slam.frontend->measures->lidar_end_time, robot_msgs::Level::OK);
#endif

                if (path_en)
                    publish_imu_path(pubImuPath, baselink_rot, baselink_pos, slam.frontend->measures->lidar_end_time);

                /******* Publish points *******/
                if (scan_pub_en)
                    if (dense_pub_en)
                        publish_cloud_world(pubLaserCloudFull, slam.feats_undistort, state, slam.frontend->measures->lidar_end_time);
                    else
                        publish_cloud(pubLaserCloudFull, slam.frontend->feats_down_world, slam.frontend->measures->lidar_end_time, map_frame);
            }
            else
            {
                LOG_ERROR("location invalid!");
#ifdef WORK
                publish_odometry2(pubOdomDev, slam.frontend->get_state(), slam.frontend->measures->lidar_end_time, slam.system_state_vaild, baselink_rot, baselink_pos);
                // publish_module_status(slam.frontend->measures->lidar_end_time, robot_msgs::Level::WARN);
#endif
#ifdef DEDUB_MODE
                publish_cloud_world(pubrelocalizationDebug, slam.frontend->measures->lidar, slam.frontend->get_state(), slam.frontend->measures->lidar_end_time);
#endif
            }

            measures_cache.pop_front();
        }
        LOG_WARN("measures cache have all been processed!");
        return;
    }
#endif

    LOG_DEBUG("run fastlio");
    QD baselink_rot;
    V3D baselink_pos;
    if (slam.run())
    {
        const auto &state = slam.frontend->get_state();
        LOG_INFO("location valid. feats_down = %d, cost time = %.1fms.", slam.frontend->loger.feats_down_size, slam.frontend->loger.total_time);
        slam.frontend->loger.print_pose(state, "cur_imu_pose");

		if (last_pose_record)
		{
			// record last pose
            fseek(last_pose_record, 0, SEEK_SET);
            fprintf(last_pose_record, "last_pose(xyz,rpy):( %.5lf %.5lf %.5lf %.5lf %.5lf %.5lf %.5lf )end!\n",
                    state.pos.x(), state.pos.y(), state.pos.z(), state.rot.w(), state.rot.x(), state.rot.y(), state.rot.z());
		}

        /******* Publish odometry *******/
#ifndef WORK
        publish_odometry(pubLidarOdom, state, slam.frontend->lidar_end_time, baselink_rot, baselink_pos);
#else
        publish_odometry2(pubOdomDev, state, slam.frontend->measures->lidar_end_time, slam.system_state_vaild, baselink_rot, baselink_pos);
        // publish_module_status(slam.frontend->measures->lidar_end_time, robot_msgs::Level::OK);
#endif

        if (path_en)
            publish_imu_path(pubImuPath, baselink_rot, baselink_pos, slam.frontend->lidar_end_time);

        /******* Publish points *******/
        if (scan_pub_en)
            if (dense_pub_en)
                publish_cloud_world(pubLaserCloudFull, slam.feats_undistort, state, slam.frontend->lidar_end_time);
            else
                publish_cloud(pubLaserCloudFull, slam.frontend->feats_down_world, slam.frontend->lidar_end_time, map_frame);
    }
    else
    {
        LOG_ERROR("location invalid!");
#ifdef WORK
        publish_odometry2(pubOdomDev, slam.frontend->get_state(), slam.frontend->measures->lidar_end_time, slam.system_state_vaild, baselink_rot, baselink_pos);
        // publish_module_status(slam.frontend->measures->lidar_end_time, robot_msgs::Level::WARN);
#endif
#ifdef DEDUB_MODE
        publish_cloud_world(pubrelocalizationDebug, slam.frontend->measures->lidar, slam.frontend->get_state(), slam.frontend->lidar_end_time);
#endif
    }
}

void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    static double last_time = 0;
    check_time_interval(last_time, msg->header.stamp.sec + msg->header.stamp.nanosec * 1.0e-9, 1.0 / slam.frontend->lidar->scan_rate, "lidar_time_interval");
    LOG_DEBUG("lidar_handler");

    Timer timer;
    pcl::PointCloud<ouster_ros::Point> pl_orig_oust;
    pcl::PointCloud<velodyne_ros::Point> pl_orig_velo;
    PointCloudType::Ptr scan(new PointCloudType());

    switch (lidar_type)
    {
    case OUST64:
        pcl::fromROSMsg(*msg, pl_orig_oust);
        slam.frontend->lidar->oust64_handler(pl_orig_oust, scan);
        break;

    case VELO16:
        pcl::fromROSMsg(*msg, pl_orig_velo);
        slam.frontend->lidar->velodyne_handler(pl_orig_velo, scan);
        break;

    default:
        printf("Error LiDAR Type");
        break;
    }

    LOG_DEBUG("lidar_cache");
    slam.frontend->cache_pointcloud_data(msg->header.stamp.sec + msg->header.stamp.nanosec * 1.0e-9, scan);
    LOG_DEBUG("lidar_process");
    sensor_data_process();
}

void livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg)
{
    static double last_time = 0;
    check_time_interval(last_time, msg->header.stamp.sec + msg->header.stamp.nanosec * 1.0e-9, 1.0 / slam.frontend->lidar->scan_rate, "lidar_time_interval");

    Timer timer;
    auto plsize = msg->point_num;
    PointCloudType::Ptr scan(new PointCloudType());
    PointCloudType::Ptr pl_orig(new PointCloudType());
    pl_orig->reserve(plsize);
    PointType point;
    for (uint i = 1; i < plsize; i++)
    {
        if (((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00))
        {
            point.x = msg->points[i].x;
            point.y = msg->points[i].y;
            point.z = msg->points[i].z;
            point.intensity = msg->points[i].reflectivity;
            point.curvature = msg->points[i].offset_time / float(1000000); // use curvature as time of each laser points, curvature unit: ms

            pl_orig->points.push_back(point);
        }
    }
    slam.frontend->lidar->avia_handler(pl_orig, scan);
    slam.frontend->cache_pointcloud_data(msg->header.stamp.sec + msg->header.stamp.nanosec * 1.0e-9, scan);
    sensor_data_process();
}

void imu_cbk(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    const auto &angular_velocity = V3D(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
    const auto &linear_acceleration = V3D(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
    if (!check_imu_meas(angular_velocity, linear_acceleration, slam.frontend->imu->imu_meas_check))
        return;

    static double last_time = 0;
    check_time_interval(last_time, msg->header.stamp.sec + msg->header.stamp.nanosec * 1.0e-9, 1.0 / slam.frontend->imu->imu_rate, "imu_time_interval");

    LOG_DEBUG("imu_cache");
    slam.frontend->cache_imu_data(msg->header.stamp.sec + msg->header.stamp.nanosec * 1.0e-9, angular_velocity, linear_acceleration);
    LOG_DEBUG("imu_process");
    sensor_data_process();
    if (slam.system_state_vaild)
    {
        auto imu_state = slam.frontend->integrate_imu_odom(*slam.frontend->imu_buffer.back());
        publish_imu_odometry(pubImuOdom, imu_state, msg->header.stamp.sec + msg->header.stamp.nanosec * 1.0e-9);
    }
}

void initialPoseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
{
    const geometry_msgs::msg::Pose &pose = msg->pose.pose;
    const auto &ori = msg->pose.pose.orientation;
    Eigen::Quaterniond quat(ori.w, ori.x, ori.y, ori.z);
    auto rpy = EigenMath::Quaternion2RPY(quat);
    // prior pose in map(imu pose)
    Pose init_pose;
    init_pose.x = pose.position.x;
    init_pose.y = pose.position.y;
    init_pose.z = pose.position.z;
    init_pose.roll = rpy.x();
    init_pose.pitch = rpy.y();
    init_pose.yaw = rpy.z();
    slam.relocalization->set_init_pose(init_pose);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("fastlio_localization");
    tf_buffer = std::make_unique<tf2_ros::Buffer>(node->get_clock());
    tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);
    tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(*node);

    string lidar_topic, imu_topic, gnss_topic;
    bool relocate_use_last_pose = true, location_log_enable = true;
    std::string last_pose_record_path;
    std::string location_log_save_path;

    load_log_parameters(node, relocate_use_last_pose, last_pose_record_path, location_log_enable, location_log_save_path);

    // 1.record last pose
	if (relocate_use_last_pose)
	{
        if (last_pose_record_path.compare("") != 0)
        {
            FileOperation::createFileWhenNotExist(last_pose_record_path);
            last_pose_record = fopen(last_pose_record_path.c_str(), "r+");
        }
		else
        {
            FileOperation::createFileWhenNotExist(DEBUG_FILE_DIR("last_pose_record.txt"));
            last_pose_record = fopen(DEBUG_FILE_DIR("last_pose_record.txt").c_str(), "r+");
        }
    }
    // 2.record log
    if (location_log_enable)
    {
        if (location_log_save_path.compare("") != 0)
            location_log = fopen(location_log_save_path.c_str(), "a");
        else
            location_log = fopen(DEBUG_FILE_DIR("location.log").c_str(), "a");
        if (location_log)
            LOG_WARN("open file %s successfully!", location_log_save_path.c_str());
        else
            LOG_ERROR("open file %s failed!", location_log_save_path.c_str());
    }
    load_ros_parameters(node, path_en, scan_pub_en, dense_pub_en, lidar_tf_broadcast, imu_tf_broadcast, lidar_topic, imu_topic, gnss_topic, map_frame, lidar_frame, baselink_frame);
    load_parameters(node, slam, lidar_type);

    /*** ROS subscribe initialization ***/
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl1;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl2;
    if (lidar_type == AVIA)
        sub_pcl1 = node->create_subscription<livox_ros_driver2::msg::CustomMsg>(lidar_topic, 1000, livox_pcl_cbk);
    else
        sub_pcl2 = node->create_subscription<sensor_msgs::msg::PointCloud2>(lidar_topic, 1000, standard_pcl_cbk);
    auto sub_imu = node->create_subscription<sensor_msgs::msg::Imu>(imu_topic, 1000, imu_cbk);
    auto sub_gnss = node->create_subscription<sensor_msgs::msg::NavSatFix>(gnss_topic, 1000, gnss_cbk);
    pubLaserCloudFull = node->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 1000);
    pubLidarOdom = node->create_publisher<nav_msgs::msg::Odometry>("/lidar_localization", 1000);
    pubImuOdom = node->create_publisher<nav_msgs::msg::Odometry>("/imu_localization", 1000);
    pubImuPath = node->create_publisher<nav_msgs::msg::Path>("/imu_path", 1000);

    auto sub_initpose = node->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>("/initialpose", 1, initialPoseCallback);
    pubrelocalizationDebug = node->create_publisher<sensor_msgs::msg::PointCloud2>("/relocalization_debug", 1);

#ifdef WORK
    pubOdomDev = node->create_publisher<localization_msg::msg::vehicle_pose>("/robot/pose_dynamic_data", 1);
    // pubModulesStatus = node->create_publisher<robot_msgs::msg::ModuleStatus>("/robot/module_status", 1);
#endif
    //------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);

    rclcpp::spin(node);

    fclose(last_pose_record);
    if (location_log_enable && location_log)
        fclose(location_log);

    return 0;
}
