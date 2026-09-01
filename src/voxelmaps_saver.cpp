#include <rclcpp/rclcpp.hpp>
#include "global_localization/bbs3d/cpu_bbs3d/bbs3d.hpp"
#include "global_localization/bbs3d/pointcloud_iof/pcd_loader_without_pcl.hpp"

#include <algorithm>
#include <vector>
#include <memory>
#include <string>


int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("fastlio_localization");

  node->declare_parameter<std::string>("map_path", "");
  node->declare_parameter<double>("min_level_res", 0.5);
  node->declare_parameter<int>("max_level", 0);

  std::string target_pcd_folder;
  double min_level_res;
  int max_level;

  node->get_parameter("map_path", target_pcd_folder);
  node->get_parameter("min_level_res", min_level_res);
  node->get_parameter("max_level", max_level);

  if (target_pcd_folder.empty()) {
    RCLCPP_ERROR(node->get_logger(), "target_pcd_folder cannot be empty");
    rclcpp::shutdown();
    return 1;
  }

  RCLCPP_INFO(node->get_logger(), "Loading target PCDs from: %s", target_pcd_folder.c_str());

  std::vector<Eigen::Vector3f> tar_points;
  float voxel_filter_width = 0.0f;
  if (!pciof::load_tar_points<float>(target_pcd_folder, voxel_filter_width, tar_points)) {
    RCLCPP_ERROR(node->get_logger(), "Failed to load target PCD files");
    rclcpp::shutdown();
    return 1;
  }

  std::vector<Eigen::Vector3d> tar_points_d;
  tar_points_d.resize(tar_points.size());
  std::transform(tar_points.begin(), tar_points.end(), tar_points_d.begin(),
                 [](const Eigen::Vector3f& p) { return p.cast<double>(); });

  RCLCPP_INFO(node->get_logger(), "Creating hierarchical voxel map...");
  auto bbs3d_ptr = std::make_unique<cpu::BBS3D>();
  bbs3d_ptr->set_tar_points(tar_points_d, min_level_res, max_level);

  RCLCPP_INFO(node->get_logger(), "Saving voxel maps...");
  bbs3d_ptr->save_voxel_params(target_pcd_folder);
  bbs3d_ptr->save_voxelmaps_pcd(target_pcd_folder);

  rclcpp::shutdown();
  return 0;
}
