#include <csignal>
#include <unistd.h>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <livox_ros_driver/CustomMsg.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include "utility/Header.h"
#include "utility/Parameters.h"
#include "system/System.hpp"


int lidar_type;
System slam;
std::string map_frame;
std::string body_frame;

bool path_en = true, scan_pub_en = false, dense_pub_en = false;
ros::Publisher pubLaserCloudFull;
ros::Publisher pubOdomAftMapped;
ros::Publisher pubImuPath;
ros::Publisher pubrelocalizationDebug;

bool flg_exit = false;
void SigHandle(int sig)
{
    flg_exit = true;
    LOG_WARN("catch sig %d", sig);
}

void publish_cloud(const ros::Publisher &pubCloud, PointCloudType::Ptr cloud, const double& lidar_end_time, const std::string& frame_id)
{
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud, cloud_msg);
    cloud_msg.header.stamp = ros::Time().fromSec(lidar_end_time);
    cloud_msg.header.frame_id = frame_id;
    pubCloud.publish(cloud_msg);
}

void publish_cloud_world(const ros::Publisher &pubLaserCloudFull, PointCloudType::Ptr laserCloud, const state_ikfom &state, const double& lidar_end_time)
{
    PointCloudType::Ptr laserCloudWorld(new PointCloudType(laserCloud->size(), 1));
    pointcloudLidarToWorld(laserCloud, laserCloudWorld, state);
    publish_cloud(pubLaserCloudFull, laserCloudWorld, lidar_end_time, map_frame);
}

template <typename T>
void set_posestamp(T &out, const state_ikfom &state)
{
    out.pose.position.x = state.pos(0);
    out.pose.position.y = state.pos(1);
    out.pose.position.z = state.pos(2);
    out.pose.orientation.x = state.rot.coeffs()[0];
    out.pose.orientation.y = state.rot.coeffs()[1];
    out.pose.orientation.z = state.rot.coeffs()[2];
    out.pose.orientation.w = state.rot.coeffs()[3];
}

void publish_tf(const geometry_msgs::Pose &pose, const double& lidar_end_time)
{
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(pose.position.x, pose.position.y, pose.position.z));
    q.setW(pose.orientation.w);
    q.setX(pose.orientation.x);
    q.setY(pose.orientation.y);
    q.setZ(pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, ros::Time().fromSec(lidar_end_time), map_frame, body_frame));
}

// 发布里程计
void publish_odometry(const ros::Publisher &pubOdomAftMapped, const state_ikfom &state, const double& lidar_end_time)
{
    nav_msgs::Odometry odomAftMapped;
    odomAftMapped.header.frame_id = map_frame;
    odomAftMapped.child_frame_id = body_frame;
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose, state);
    pubOdomAftMapped.publish(odomAftMapped);
    publish_tf(odomAftMapped.pose.pose, lidar_end_time);
}

void publish_imu_path(const ros::Publisher &pubPath, const state_ikfom &state, const double& lidar_end_time)
{
    static geometry_msgs::PoseStamped msg_body_pose;
    set_posestamp(msg_body_pose, state);
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

void sensor_data_process()
{
    if (flg_exit)
        return;

    if (!slam.frontend->sync_sensor_data())
        return;

    if (slam.run())
    {
        const auto &state = slam.frontend->get_state();

        /******* Publish odometry *******/
        publish_odometry(pubOdomAftMapped, state, slam.frontend->lidar_end_time);

        /******* Publish points *******/
        if (path_en)
            publish_imu_path(pubImuPath, state, slam.frontend->lidar_end_time);
        if (scan_pub_en)
            if (dense_pub_en)
                publish_cloud_world(pubLaserCloudFull, slam.feats_undistort, state, slam.frontend->lidar_end_time);
            else
                publish_cloud(pubLaserCloudFull, slam.frontend->feats_down_world, slam.frontend->lidar_end_time, map_frame);
    }
    else
    {
#ifdef DEDUB_MODE
        publish_cloud_world(pubrelocalizationDebug, slam.frontend->measures->lidar, slam.frontend->get_state(), slam.lidar_end_time);
#endif
    }
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    static double last_time = 0;
    check_time_interval(last_time, msg->header.stamp.toSec(), 1.0 / slam.frontend->lidar->scan_rate, "lidar");

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

    slam.frontend->cache_pointcloud_data(msg->header.stamp.toSec(), scan);
    slam.frontend->loger.preprocess_time = timer.elapsedStart();
    sensor_data_process();
}

void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
    static double last_time = 0;
    check_time_interval(last_time, msg->header.stamp.toSec(), 1.0 / slam.frontend->lidar->scan_rate, "lidar");

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
    slam.frontend->loger.preprocess_time = timer.elapsedStart();
    sensor_data_process();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg)
{
    static double last_time = 0;
    check_time_interval(last_time, msg->header.stamp.toSec(), 1.0 / slam.frontend->imu->imu_rate, "imu");

    slam.frontend->cache_imu_data(msg->header.stamp.toSec(),
                                  V3D(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z),
                                  V3D(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z),
                                  QD(msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z));
    sensor_data_process();
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
    init_pose.roll = rpy.x();
    init_pose.pitch = rpy.y();
    init_pose.yaw = rpy.z();
    slam.relocalization->set_init_pose(init_pose);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "SLAM");
    ros::NodeHandle nh;
    string lidar_topic, imu_topic, config_file;

    ros::param::param("config_file", config_file, std::string(""));
    load_ros_parameters(string(ROOT_DIR) + config_file, path_en, scan_pub_en, dense_pub_en, lidar_topic, imu_topic, map_frame, body_frame);
    load_parameters(slam, string(ROOT_DIR) + config_file, lidar_type);

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = lidar_type == AVIA ? nh.subscribe(lidar_topic, 200000, livox_pcl_cbk) : nh.subscribe(lidar_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    pubImuPath = nh.advertise<nav_msgs::Path>("/imu_path", 100000);

    ros::Subscriber sub_initpose = nh.subscribe("/initialpose", 1, initialPoseCallback);
    pubrelocalizationDebug = nh.advertise<sensor_msgs::PointCloud2>("/relocalization_debug", 1);
    //------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);

    ros::spin();

    return 0;
}
