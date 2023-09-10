#include <csignal>
#include <unistd.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/transform_broadcaster.h>

#include "livox_interfaces2/msg/custom_msg.hpp"
// #include <livox_ros_driver/CustomMsg.h>
#include "utility/Header.h"
#include "utility/Parameters.h"
#include "system/System.hpp"

int lidar_type;
System slam;
std::string map_frame;
std::string body_frame;

bool flg_exit = false;
void SigHandle(int sig)
{
    flg_exit = true;
    LOG_WARN("catch sig %d", sig);
}

void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    Timer timer;
    pcl::PointCloud<ouster_ros::Point> pl_orig_oust;
    pcl::PointCloud<velodyne_ros::Point> pl_orig_velo;
    PointCloudType::Ptr scan(new PointCloudType());

    switch (lidar_type)
    {
    case OUST64:
        pcl::fromROSMsg(*msg, pl_orig_oust);
        slam.lidar->oust64_handler(pl_orig_oust, scan);
        break;

    case VELO16:
        pcl::fromROSMsg(*msg, pl_orig_velo);
        slam.lidar->velodyne_handler(pl_orig_velo, scan);
        break;

    default:
        printf("Error LiDAR Type");
        break;
    }

    slam.cache_pointcloud_data(msg->header.stamp.sec + msg->header.stamp.nanosec * 1.0e-9, scan);
    slam.loger.preprocess_time = timer.elapsedStart();
}

void livox_pcl_cbk(const livox_interfaces2::msg::CustomMsg::SharedPtr msg)
{
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
    slam.lidar->avia_handler(pl_orig, scan);
    slam.cache_pointcloud_data(msg->header.stamp.sec + msg->header.stamp.nanosec * 1.0e-9, scan);
    slam.loger.preprocess_time = timer.elapsedStart();
}

void imu_cbk(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    slam.cache_imu_data(msg->header.stamp.sec + msg->header.stamp.nanosec * 1.0e-9,
                        V3D(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z),
                        V3D(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z));
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
    PointCloudType::Ptr laserCloudWorld(new PointCloudType);
    pointcloudLidarToWorld(laserCloud, laserCloudWorld, state);
    publish_cloud(pubLaserCloudFull, laserCloudWorld, lidar_end_time, map_frame);
}

// 发布ikd-tree地图
void publish_ikdtree_map(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubLaserCloudMap, PointCloudType::Ptr featsFromMap, const double& lidar_end_time)
{
    publish_cloud(pubLaserCloudMap, featsFromMap, lidar_end_time, map_frame);
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

// 发布里程计
void publish_odometry(rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr &pubOdomAftMapped, tf2_ros::TransformBroadcaster &broadcaster,
                      const state_ikfom &state, const esekfom::esekf<state_ikfom, 12, input_ikfom> &kf, const double &lidar_end_time)
{
    static nav_msgs::msg::Odometry odomAftMapped;
    odomAftMapped.header.frame_id = map_frame;
    odomAftMapped.child_frame_id = body_frame;
    odomAftMapped.header.stamp = rclcpp::Time(lidar_end_time * 1e9);
    set_posestamp(odomAftMapped.pose, state);
    pubOdomAftMapped->publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        // 设置协方差 P里面先是旋转后是位置 这个POSE里面先是位置后是旋转 所以对应的协方差要对调一下
        odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
    }

    static geometry_msgs::msg::TransformStamped transform_stamped;
    transform_stamped.header.stamp = odomAftMapped.header.stamp;
    transform_stamped.header.frame_id = map_frame;
    transform_stamped.child_frame_id = body_frame;
    transform_stamped.transform.translation.x = odomAftMapped.pose.pose.position.x;
    transform_stamped.transform.translation.y = odomAftMapped.pose.pose.position.y;
    transform_stamped.transform.translation.z = odomAftMapped.pose.pose.position.z;
    transform_stamped.transform.rotation.x = odomAftMapped.pose.pose.orientation.x;
    transform_stamped.transform.rotation.y = odomAftMapped.pose.pose.orientation.y;
    transform_stamped.transform.rotation.z = odomAftMapped.pose.pose.orientation.z;
    transform_stamped.transform.rotation.w = odomAftMapped.pose.pose.orientation.w;
    broadcaster.sendTransform(transform_stamped);
}

// 每隔10个发布一下轨迹
void publish_imu_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr &pubPath, const state_ikfom &state, const double& lidar_end_time)
{
    static geometry_msgs::msg::PoseStamped msg_body_pose;
    set_posestamp(msg_body_pose, state);
    msg_body_pose.header.stamp = rclcpp::Time(lidar_end_time * 1e9);
    msg_body_pose.header.frame_id = map_frame;

    static nav_msgs::msg::Path path;
    path.header.stamp = msg_body_pose.header.stamp;
    path.header.frame_id = map_frame;

    path.poses.push_back(msg_body_pose);
    pubPath->publish(path);
    /*** if path is too large, the rvis will crash ***/
    if (path.poses.size() >= 300)
    {
        path.poses.erase(path.poses.begin());
    }
}


int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("SLAM");
    bool pure_localization = false;
    bool save_globalmap_en = false, path_en = true;
    bool scan_pub_en = false, dense_pub_en = false;
    string lidar_topic, imu_topic, config_file;

    ros::param::param("config_file", config_file, std::string(""));

    load_ros_parameters(string(ROOT_DIR) + config_file, path_en, scan_pub_en, dense_pub_en, lidar_topic, imu_topic, map_frame, body_frame);
    load_parameters(slam, string(ROOT_DIR) + config_file, pure_localization, save_globalmap_en, lidar_type);

    /*** ROS subscribe initialization ***/
    rclcpp::Subscription<livox_interfaces2::msg::CustomMsg>::SharedPtr sub_pcl1;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl2;
    if (lidar_type == AVIA)
        sub_pcl1 = node->create_subscription<livox_interfaces2::msg::CustomMsg>(lidar_topic, 200000, livox_pcl_cbk);
    else
        sub_pcl2 = node->create_subscription<sensor_msgs::msg::PointCloud2>(lidar_topic, 200000, standard_pcl_cbk);
    auto sub_imu = node->create_subscription<sensor_msgs::msg::Imu>(imu_topic, 200000, imu_cbk);
    // 发布当前正在扫描的点云，topic名字为/cloud_registered
    auto pubLaserCloudFull = node->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 100000);
    // not used
    auto pubLaserCloudEffect = node->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_effected", 100000);
    // not used
    auto pubLaserCloudMap = node->create_publisher<sensor_msgs::msg::PointCloud2>("/Laser_map", 100000);
    auto pubOdomAftMapped = node->create_publisher<nav_msgs::msg::Odometry>("/Odometry", 100000);
    auto pubImuPath = node->create_publisher<nav_msgs::msg::Path>("/imu_path", 100000);
    tf2_ros::TransformBroadcaster broadcaster(node);
    //------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    rclcpp::Rate rate(5000);
    while (rclcpp::ok())
    {
        if (flg_exit)
            break;
        rclcpp::spin_some(node);

        if (!slam.sync_sensor_data())
            continue;

        if (slam.run())
        {
            const auto &state = slam.frontend->state;

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped, broadcaster, state, slam.frontend->kf, slam.lidar_end_time);

            /******* Publish points *******/
            if (path_en)
                publish_imu_path(pubImuPath, state, slam.lidar_end_time);
            if (scan_pub_en)
                if (dense_pub_en)
                    publish_cloud_world(pubLaserCloudFull, slam.feats_undistort, state, slam.lidar_end_time);
                else
                    publish_cloud_world(pubLaserCloudFull, slam.frontend->feats_down_lidar, state, slam.lidar_end_time);

            // publish_cloud_world(pubLaserCloudEffect, laserCloudOri, state, slam.lidar_end_time);
            if (0)
            {
                PointCloudType::Ptr featsFromMap(new PointCloudType());
                slam.frontend->get_ikdtree_point(featsFromMap);
                publish_ikdtree_map(pubLaserCloudMap, featsFromMap, slam.lidar_end_time);
            }
        }

        rate.sleep();
    }

    return 0;
}
