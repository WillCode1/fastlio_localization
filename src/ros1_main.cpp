#include <csignal>
#include <unistd.h>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/OccupancyGrid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <livox_ros_driver/CustomMsg.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include "system/Header.h"
#include "system/ParametersRos1.h"
#include "system/System.hpp"
#include "slam_interfaces/InsPvax.h"
#include <chcnav/hcinspvatzcb.h>
// #define EVO
// #define UrbanLoco
// #define liosam

#define MEASURES_BUFFER
// #define WORK
#ifdef WORK
#include "slam_interfaces/VehiclePose.h"
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

#ifdef EVO
FILE *file_pose_fastlio_lidar;
FILE *file_pose_fastlio_imu;
FILE *file_pose_gnss;
#endif

double lidar_turnover_roll, lidar_turnover_pitch;
bool path_en = true, scan_pub_en = false, dense_pub_en = false, lidar_tf_broadcast = false, imu_tf_broadcast = false;
ros::Publisher pubLaserCloudFull;
ros::Publisher pubLidarOdom;
ros::Publisher pubImuOdom;
ros::Publisher pubImuPath;
ros::Publisher pubrelocalizationDebug;
ros::Publisher pubGlobalMap;
ros::Publisher pubOccGrid;
#ifdef WORK
ros::Publisher pubOdomDev;
ros::Publisher pubModulesStatus;
#endif

bool flg_exit = false;
void SigHandle(int sig)
{
    flg_exit = true;
    LOG_WARN("catch sig %d", sig);
}

void publish_cloud(const ros::Publisher &pubCloud, PointCloudType::Ptr cloud, const double &lidar_end_time, const std::string &frame_id)
{
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud, cloud_msg);
    cloud_msg.header.stamp = ros::Time().fromSec(lidar_end_time);
    cloud_msg.header.frame_id = frame_id;
    pubCloud.publish(cloud_msg);
}

void publish_cloud_world(const ros::Publisher &pubLaserCloudFull, PointCloudType::Ptr laserCloud, const state_ikfom &state, const double &lidar_end_time)
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
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(pos.x(), pos.y(), pos.z()));
    q.setW(rot.w());
    q.setX(rot.x());
    q.setY(rot.y());
    q.setZ(rot.z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, ros::Time().fromSec(lidar_end_time), map_frame, baselink_frame));
}

void publish_odometry(const ros::Publisher &pubLidarOdom, const state_ikfom &state, const double &lidar_end_time, QD &baselink_rot, V3D &baselink_pos)
{
    static bool first_time = true;
    static Eigen::Quaterniond lidar2baselink_q;
    static Eigen::Vector3d lidar2baselink_t;
    if (first_time)
    {
        first_time = false;
        tf::TransformListener tfListener;
        tf::StampedTransform trans_lidar2baselink;
        trans_lidar2baselink.setIdentity();

        try
        {
            tfListener.waitForTransform(lidar_frame, baselink_frame, ros::Time(0), ros::Duration(3.0));
            tfListener.lookupTransform(lidar_frame, baselink_frame, ros::Time(0), trans_lidar2baselink);
        }
        catch (tf::TransformException &ex)
        {
            ROS_ERROR("%s", ex.what());
        }

        auto pos = trans_lidar2baselink.getOrigin();
        auto quat = trans_lidar2baselink.getRotation();
        lidar2baselink_q = Eigen::Quaterniond(quat.w(), quat.x(), quat.y(), quat.z());
        lidar2baselink_t = Eigen::Vector3d(pos.x(), pos.y(), pos.z());
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
    auto baselink2imu = lidar2baselink.inverse() * lidar2imu;
    pose_mat = pose_mat * baselink2imu;
    baselink_rot = QD(M3D(pose_mat.topLeftCorner(3, 3)));
    baselink_pos = pose_mat.topRightCorner(3, 1);

    if (std::abs(baselink_pos.x()) > 1e7 || std::abs(baselink_pos.y()) > 1e7 || std::abs(baselink_pos.z()) > 1e7)
    {
        LOG_WARN("localization state maybe abnormal! (baselink frame) pos(%f, %f, %f)", baselink_pos.x(), baselink_pos.y(), baselink_pos.z());
    }

#ifdef EVO
    LogAnalysis::save_trajectory(file_pose_fastlio_lidar, baselink_pos, baselink_rot, lidar_end_time);
#endif

    nav_msgs::Odometry odomAftMapped;
    odomAftMapped.header.frame_id = map_frame;
    odomAftMapped.child_frame_id = baselink_frame;
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose, baselink_rot, baselink_pos);
    pubLidarOdom.publish(odomAftMapped);
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

void publish_odometry2(const ros::Publisher &pubOdomDev, const state_ikfom &state, const double &lidar_end_time, bool vaild, QD &baselink_rot, V3D &baselink_pos)
{
    slam_interfaces::VehiclePose odom;
    odom.header.frame_id = map_frame;
    odom.header.stamp = ros::Time().fromSec(lidar_end_time);
    odom.is_valid = vaild;

    static bool first_time = true;
    static Eigen::Quaterniond lidar2baselink_q;
    static Eigen::Vector3d lidar2baselink_t;
    if (first_time)
    {
        first_time = false;
        tf::TransformListener tfListener;
        tf::StampedTransform trans_lidar2baselink;
        trans_lidar2baselink.setIdentity();

        try
        {
            tfListener.waitForTransform(lidar_frame, baselink_frame, ros::Time(0), ros::Duration(3.0));
            tfListener.lookupTransform(lidar_frame, baselink_frame, ros::Time(0), trans_lidar2baselink);
        }
        catch (tf::TransformException ex)
        {
            ROS_ERROR("%s", ex.what());
        }

        auto pos = trans_lidar2baselink.getOrigin();
        auto quat = trans_lidar2baselink.getRotation();
        lidar2baselink_q = Eigen::Quaterniond(quat.w(), quat.x(), quat.y(), quat.z());
        lidar2baselink_t = Eigen::Vector3d(pos.x(), pos.y(), pos.z());
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
    auto baselink2imu = lidar2baselink.inverse() * lidar2imu;
    pose_mat = pose_mat * baselink2imu;
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

void publish_imu_odometry(const ros::Publisher &pubImuOdom, const state_ikfom &state, const double &imu_time)
{
    static bool first_time = true;
    static Eigen::Quaterniond lidar2baselink_q;
    static Eigen::Vector3d lidar2baselink_t;
    if (first_time)
    {
        first_time = false;
        tf::TransformListener tfListener;
        tf::StampedTransform trans_lidar2baselink;
        trans_lidar2baselink.setIdentity();

        try
        {
            tfListener.waitForTransform(lidar_frame, baselink_frame, ros::Time(0), ros::Duration(3.0));
            tfListener.lookupTransform(lidar_frame, baselink_frame, ros::Time(0), trans_lidar2baselink);
        }
        catch (tf::TransformException &ex)
        {
            ROS_ERROR("%s", ex.what());
        }

        auto pos = trans_lidar2baselink.getOrigin();
        auto quat = trans_lidar2baselink.getRotation();
        lidar2baselink_q = Eigen::Quaterniond(quat.w(), quat.x(), quat.y(), quat.z());
        lidar2baselink_t = Eigen::Vector3d(pos.x(), pos.y(), pos.z());
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
    auto baselink2imu = lidar2baselink.inverse() * lidar2imu;
    pose_mat = pose_mat * baselink2imu;
    QD baselink_rot = QD(M3D(pose_mat.topLeftCorner(3, 3)));
    V3D baselink_pos = pose_mat.topRightCorner(3, 1);

    if (std::abs(baselink_pos.x()) > 1e7 || std::abs(baselink_pos.y()) > 1e7 || std::abs(baselink_pos.z()) > 1e7)
    {
        LOG_WARN("localization state maybe abnormal! (baselink frame) pos(%f, %f, %f)", baselink_pos.x(), baselink_pos.y(), baselink_pos.z());
    }

#ifdef EVO
    LogAnalysis::save_trajectory(file_pose_fastlio_imu, baselink_pos, baselink_rot, imu_time);
#endif

    nav_msgs::Odometry imuOdom;
    imuOdom.header.frame_id = map_frame;
    imuOdom.child_frame_id = baselink_frame;
    imuOdom.header.stamp = ros::Time().fromSec(imu_time);
    set_posestamp(imuOdom.pose, baselink_rot, baselink_pos);
    pubImuOdom.publish(imuOdom);
    if (imu_tf_broadcast)
        publish_tf(baselink_rot, baselink_pos, imu_time);
}

void publish_imu_path(const ros::Publisher &pubPath, const QD &rot, const V3D &pos, const double &lidar_end_time)
{
    static geometry_msgs::PoseStamped msg_body_pose;
    set_posestamp(msg_body_pose, rot, pos);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = map_frame;

    static nav_msgs::Path path;
    path.header.stamp = ros::Time::now();
    path.header.frame_id = map_frame;

    path.poses.push_back(msg_body_pose);
    pubPath.publish(path);
    /*** if path is too large, the rvis will crash ***/
    if (path.poses.size() >= 300)
    {
        path.poses.erase(path.poses.begin());
    }
}

void chcnav_cbk(const chcnav::hcinspvatzcb::ConstPtr &msg)
{
    V3D gnss_position = V3D(msg->latitude, msg->longitude, msg->altitude);
    QD rot = EigenMath::RPY2Quaternion(V3D(DEG2RAD(msg->roll), DEG2RAD(msg->pitch), DEG2RAD(msg->yaw)));
#ifdef ENU
    gnss_position = enu_coordinate::Earth::LLH2ENU(gnss_position, true);
#else
    gnss_position = utm_coordinate::LLAtoUTM2(gnss_position);
#endif
    slam.relocalization->gnss_pose = GnssPose(msg->header.stamp.toSec(), gnss_position, rot);
#ifdef EVO
    LogAnalysis::save_trajectory(file_pose_gnss, gnss_position, rot, msg->header.stamp.toSec());
#endif
}

void gnss_cbk(const slam_interfaces::InsPvax::ConstPtr &msg)
{
    V3D gnss_position = V3D(RAD2DEG(msg->latitude), RAD2DEG(msg->longitude), msg->altitude);
    QD rot = EigenMath::RPY2Quaternion(V3D(msg->roll, msg->pitch, msg->azimuth));
#ifdef ENU
    gnss_position = enu_coordinate::Earth::LLH2ENU(gnss_position, true);
#else
    gnss_position = utm_coordinate::LLAtoUTM2(gnss_position);
#endif
    slam.relocalization->gnss_pose = GnssPose(msg->header.stamp.toSec(), gnss_position, rot);
#ifdef EVO
    LogAnalysis::save_trajectory(file_pose_gnss, gnss_position, rot, msg->header.stamp.toSec());
#endif
}

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

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    static double last_time = 0;
    check_time_interval(last_time, msg->header.stamp.toSec(), 1.0 / slam.frontend->lidar->scan_rate, "lidar_time_interval");
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
    slam.frontend->cache_pointcloud_data(msg->header.stamp.toSec(), scan);
    LOG_DEBUG("lidar_process");
    sensor_data_process();
}

void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
    static double last_time = 0;
    check_time_interval(last_time, msg->header.stamp.toSec(), 1.0 / slam.frontend->lidar->scan_rate, "lidar_time_interval");

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
    slam.frontend->cache_pointcloud_data(msg->header.stamp.toSec(), scan);
    sensor_data_process();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg)
{
    const auto &angular_velocity = V3D(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
    const auto &linear_acceleration = V3D(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
    if (!check_imu_meas(angular_velocity, linear_acceleration, slam.frontend->imu->imu_meas_check))
        return;

    static double last_time = 0;
    check_time_interval(last_time, msg->header.stamp.toSec(), 1.0 / slam.frontend->imu->imu_rate, "imu_time_interval");

    LOG_DEBUG("imu_cache");
    slam.frontend->cache_imu_data(msg->header.stamp.toSec(), angular_velocity, linear_acceleration);
    LOG_DEBUG("imu_process");
    sensor_data_process();
    if (slam.system_state_vaild)
    {
        auto imu_state = slam.frontend->integrate_imu_odom(*slam.frontend->imu_buffer.back());
        publish_imu_odometry(pubImuOdom, imu_state, msg->header.stamp.toSec());
    }
}

void publish_global_map(const ros::TimerEvent &)
{
    if (pubGlobalMap.getNumSubscribers() != 0)
    {
        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(*slam.global_map, cloud_msg);
        cloud_msg.header.stamp = ros::Time::now();
        cloud_msg.header.frame_id = map_frame;
        pubGlobalMap.publish(cloud_msg);
    }

    if (pubOccGrid.getNumSubscribers() != 0 && !slam.p2p->map_data.empty())
    {
        nav_msgs::OccupancyGrid msg;
        msg.header.stamp = ros::Time::now();
        msg.header.frame_id = map_frame;

        msg.info.map_load_time = ros::Time::now();
        msg.info.resolution = slam.p2p->resolution_;

        msg.info.origin.position.x = slam.p2p->map_info_origin_position_x;
        msg.info.origin.position.y = slam.p2p->map_info_origin_position_y;
        msg.info.origin.orientation.w = 1.0;

        msg.info.width = slam.p2p->map_info_width;
        msg.info.height = slam.p2p->map_info_height;
        msg.data = slam.p2p->map_data;
        pubOccGrid.publish(msg);
    }
}

void initialPoseCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr &msg)
{
    const geometry_msgs::Pose &pose = msg->pose.pose;
    const auto &ori = msg->pose.pose.orientation;
    Eigen::Quaterniond quat(ori.w, ori.x, ori.y, ori.z);
    auto rpy = EigenMath::Quaternion2RPY(quat);
    // prior pose in map(imu pose)
    Pose init_pose;
    init_pose.x = pose.position.x;
    init_pose.y = pose.position.y;
    init_pose.z = pose.position.z;
    init_pose.roll = DEG2RAD(lidar_turnover_roll);
    init_pose.pitch = DEG2RAD(lidar_turnover_pitch);
    init_pose.yaw = rpy.z();
    slam.relocalization->set_init_pose(init_pose);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "SLAM");
    ros::NodeHandle nh;
    string lidar_topic, imu_topic, gnss_topic;
    bool relocate_use_last_pose = true, location_log_enable = true;
    std::string last_pose_record_path;
    std::string location_log_save_path;
    std::vector<double> lla;

#ifdef EVO
    file_pose_fastlio_lidar = fopen(DEBUG_FILE_DIR("fastlio_pose.txt").c_str(), "w");
    fprintf(file_pose_fastlio_lidar, "# fastlio_pose trajectory\n# timestamp tx ty tz qx qy qz qw\n");
    file_pose_fastlio_imu = fopen(DEBUG_FILE_DIR("fastlio_pose_imu.txt").c_str(), "w");
    fprintf(file_pose_fastlio_imu, "# fastlio_pose_imu trajectory\n# timestamp tx ty tz qx qy qz qw\n");
    file_pose_gnss = fopen(DEBUG_FILE_DIR("gnss_pose.txt").c_str(), "w");
    fprintf(file_pose_gnss, "# gnss trajectory\n# timestamp tx ty tz qx qy qz qw\n");
#endif

    ros::param::param("relocalization_cfg/lidar_turnover_roll", lidar_turnover_roll, 0.);
    ros::param::param("relocalization_cfg/lidar_turnover_pitch", lidar_turnover_pitch, 0.);
    ros::param::param("official/lla", lla, std::vector<double>());

    load_log_parameters(relocate_use_last_pose, last_pose_record_path, location_log_enable, location_log_save_path);

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
    load_ros_parameters(path_en, scan_pub_en, dense_pub_en, lidar_tf_broadcast, imu_tf_broadcast, lidar_topic, imu_topic, gnss_topic, map_frame, lidar_frame, baselink_frame);
    load_parameters(slam, lidar_type);

    if (lla.size() == 3 && lla[0] != 0)
    {
#ifdef ENU
        enu_coordinate::Earth::SetOrigin(V3D(lla[0], lla[1], lla[2]));
#else
        utm_coordinate::SetUtmOrigin(V3D(lla[0], lla[1], lla[2]));
#endif
    }
    else
    {
        LOG_WARN("use gps, please set the origin first!");
    }

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = lidar_type == AVIA ? nh.subscribe(lidar_topic, 200000, livox_pcl_cbk) : nh.subscribe(lidar_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    ros::Subscriber sub_gnss = nh.subscribe(gnss_topic, 200000, chcnav_cbk);
    pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    pubLidarOdom = nh.advertise<nav_msgs::Odometry>("/lidar_localization", 100000);
    pubImuOdom = nh.advertise<nav_msgs::Odometry>("/imu_localization", 100000);
    pubImuPath = nh.advertise<nav_msgs::Path>("/imu_path", 100000);
    pubGlobalMap = nh.advertise<sensor_msgs::PointCloud2>("/global_map", 1);
    pubOccGrid = nh.advertise<nav_msgs::OccupancyGrid>("/grid_map", 1);

    ros::Timer timer = nh.createTimer(ros::Duration(2.0), publish_global_map);
    ros::Subscriber sub_initpose = nh.subscribe("/initialpose", 1, initialPoseCallback);
    pubrelocalizationDebug = nh.advertise<sensor_msgs::PointCloud2>("/relocalization_debug", 1);

#ifdef WORK
    pubOdomDev = nh.advertise<slam_interfaces::VehiclePose>("/robot/pose_dynamic_data", 1);
    // pubModulesStatus = nh.advertise<robot_msgs::ModuleStatus>("/robot/module_status", 1);
#endif
    //------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);

    ros::spin();

	if (relocate_use_last_pose)
        fclose(last_pose_record);
    if (location_log_enable && location_log)
        fclose(location_log);

    return 0;
}
