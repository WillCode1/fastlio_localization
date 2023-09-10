#include <pcl/registration/gicp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include "system/DataDef.h"

std::vector<float> trans_state = std::vector<float>(6, 0);
std::vector<float> offset = std::vector<float>(6, 0.1);
std::vector<float> trans_state_backup;
int trans_state_index = 0;
int offset_index = 0;

double getFitnessScore(PointCloudType::Ptr submap, PointCloudType::Ptr scan, const Eigen::Affine3f &trans, double max_range)
{
  double fitness_score = 0.0;

  // Transform the input dataset using the final transformation
  PointCloudType::Ptr scan_tmp(new PointCloudType);
  pcl::transformPointCloud(*scan, *scan_tmp, trans);

  pcl::search::KdTree<PointType> kdtree;
  kdtree.setInputCloud(submap);

  std::vector<int> nn_indices(1);
  std::vector<float> nn_dists(1);

  // For each point in the source dataset
  int nr = 0;
  for (size_t i = 0; i < scan_tmp->points.size(); ++i)
  {
      // Find its nearest neighbor in the target
      kdtree.nearestKSearch(scan_tmp->points[i], 1, nn_indices, nn_dists);

      // Deal with occlusions (incomplete targets)
      if (nn_dists[0] <= max_range)
      {
          // Add to the fitness score
          fitness_score += nn_dists[0];
          nr++;
      }
  }

  if (nr > 0)
      return (fitness_score / nr);
  else
      return (std::numeric_limits<double>::max());
}

// 键盘事件回调函数
void keyboardEventCallback(const pcl::visualization::KeyboardEvent &event, void *viewer_void)
{
    pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *>(viewer_void);

    if (event.getKeySym() == "q" && event.keyDown())
    {
        viewer->close();
    }
    else if (event.getKeySym() == "r" && event.keyDown())
    {
        trans_state = trans_state_backup;
    }
    else if (event.getKeySym() == "s" && event.keyDown())
    {
        viewer->saveScreenshot(string(ROOT_DIR) + "/screenshot.png");
    }
    else if (event.getKeySym() == "exclam" && event.keyDown())
    {
        offset_index = 0;
    }
    else if (event.getKeySym() == "at" && event.keyDown())
    {
        offset_index = 1;
    }
    else if (event.getKeySym() == "numbersign" && event.keyDown())
    {
        offset_index = 2;
    }
    else if (event.getKeySym() == "dollar" && event.keyDown())
    {
        offset_index = 3;
    }
    else if (event.getKeySym() == "percent" && event.keyDown())
    {
        offset_index = 4;
    }
    else if (event.getKeySym() == "asciicircum" && event.keyDown())
    {
        offset_index = 5;
    }
    else if (event.getKeySym() == "1")
    {
        trans_state_index = 0;
    }
    else if (event.getKeySym() == "2")
    {
        trans_state_index = 1;
    }
    else if (event.getKeySym() == "3")
    {
        trans_state_index = 2;
    }
    else if (event.getKeySym() == "4")
    {
        trans_state_index = 3;
    }
    else if (event.getKeySym() == "5")
    {
        trans_state_index = 4;
    }
    else if (event.getKeySym() == "6")
    {
        trans_state_index = 5;
    }
    else if (event.getKeySym() == "Up" && event.keyDown() && event.isShiftPressed())
    {
        offset[offset_index] += 0.01;
    }
    else if (event.getKeySym() == "Down" && event.keyDown() && event.isShiftPressed())
    {
        offset[offset_index] -= 0.01;
    }
    else if (event.getKeySym() == "Up" && event.keyDown())
    {
        trans_state[trans_state_index] += offset[trans_state_index];
    }
    else if (event.getKeySym() == "Down" && event.keyDown())
    {
        trans_state[trans_state_index] -= offset[trans_state_index];
    }
}

double manually_adjust_loop_closure(PointCloudType::Ptr submap, PointCloudType::Ptr scan, Eigen::Affine3f& transform)
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));

    // backup
    trans_state_backup = trans_state;

    transform = pcl::getTransformation(trans_state[0], trans_state[1], trans_state[2], trans_state[3], trans_state[4], trans_state[5]);
    PointCloudType::Ptr transformedCloud(new PointCloudType);
    pcl::transformPointCloud(*scan, *transformedCloud, transform);

    // 添加点云到可视化窗口
    viewer->addPointCloud<PointType>(submap, "submap");
    viewer->addPointCloud<PointType>(transformedCloud, "scan");

    // 设置点云渲染属性
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0, "submap");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "scan");

    char buffer[1024];
    sprintf(buffer, "current_pose (x=%f, y=%f, z=%f, roll=%f, pitch=%f, yaw=%f), offset (x=%.2f, y=%.2f, z=%.2f, roll=%.3f, pitch=%.3f, yaw=%.3f)",
            trans_state[0], trans_state[1], trans_state[2], trans_state[3], trans_state[4], trans_state[5], offset[0], offset[1], offset[2], offset[3], offset[4], offset[5]);
    viewer->addText(buffer, 50, 50, "text_1");

    // 注册键盘事件回调函数
    viewer->registerKeyboardCallback(keyboardEventCallback, viewer.get());

    double score = 0;
    // 循环显示
    while (!viewer->wasStopped())
    {
        score = getFitnessScore(submap, scan, transform, 2);
        sprintf(buffer, "current_pose (x=%.3f, y=%.3f, z=%.3f, roll=%.3f, pitch=%.3f, yaw=%.3f), offset (x=%.2f, y=%.2f, z=%.2f, roll=%.3f, pitch=%.3f, yaw=%.3f), config=(%d, %d), FitnessScore=%lf",
                trans_state[0], trans_state[1], trans_state[2], trans_state[3], trans_state[4], trans_state[5],
                offset[0], offset[1], offset[2], offset[3], offset[4], offset[5], trans_state_index + 1, offset_index + 1, score);
        viewer->updateText(buffer, 50, 50, "text_1");

        transform = pcl::getTransformation(trans_state[0], trans_state[1], trans_state[2], trans_state[3], trans_state[4], trans_state[5]);
        pcl::transformPointCloud(*scan, *transformedCloud, transform);
        viewer->updatePointCloud<PointType>(transformedCloud, "scan");

        viewer->spinOnce();
    }

    viewer->close();
    return score;
}
