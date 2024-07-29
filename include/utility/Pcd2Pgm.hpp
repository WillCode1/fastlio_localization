/*
 * map_saver
 * Copyright (c) 2008, Willow Garage, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the <ORGANIZATION> nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * @ref: https://github.com/ros-planning/navigation/tree/kinetic-devel/map_server/src
 */
#pragma once
#include <cstdio>
#include <fstream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

using namespace std;
using PointType = pcl::PointXYZINormal;
using PointCloudType = pcl::PointCloud<PointType>;

/**
 * @brief Map generation node.
 */
class Pcd2Pgm
{
public:
  Pcd2Pgm(const double &map_resolution, const std::string &save_path)
      : resolution_(map_resolution), save_path_(save_path) {}

  void convert_from_pcd(const PointCloudType::Ptr cloud)
  {
    double x_min, x_max, y_min, y_max; // 这里是投影到xy平面，如果要投到xz/yz，这里以及后面的xy对应的数据改为你想投影的平面

    if (cloud->points.empty())
    {
      printf("pcd is empty!");
      return;
    }

    for (auto i = 0; i < cloud->points.size() - 1; i++)
    {
      if (i == 0)
      {
        x_min = x_max = cloud->points[i].x;
        y_min = y_max = cloud->points[i].y;
      }

      double x = cloud->points[i].x;
      double y = cloud->points[i].y;

      if (x < x_min)
        x_min = x;
      if (x > x_max)
        x_max = x;

      if (y < y_min)
        y_min = y;
      if (y > y_max)
        y_max = y;
    }

    map_info_origin_position_x = x_min;
    map_info_origin_position_y = y_min;

    map_info_width = int((x_max - x_min) / resolution_);
    map_info_height = int((y_max - y_min) / resolution_);

    map_data.resize(map_info_width * map_info_height);
    map_data.assign(map_info_width * map_info_height, 0);

    for (auto iter = 0; iter < cloud->points.size(); iter++)
    {
      int i = int((cloud->points[iter].x - x_min) / resolution_);
      if (i < 0 || i >= map_info_width)
        continue;

      int j = int((cloud->points[iter].y - y_min) / resolution_);
      if (j < 0 || j >= map_info_height - 1)
        continue;

      map_data[i + j * map_info_width] = 100;
    }
  }

  void convert_to_pgm()
  {
    printf("Received a %d X %d map @ %.3f m/pix\n",
           map_info_width, map_info_height, resolution_);

    std::string mapdatafile = save_path_ + ".pgm";
    printf("Writing map occupancy data to %s\n", mapdatafile.c_str());
    FILE *out = fopen(mapdatafile.c_str(), "w");
    if (!out)
    {
      printf("Couldn't save map file to %s\n", mapdatafile.c_str());
      return;
    }

    fprintf(out, "P5\n%d %d\n255\n", map_info_width, map_info_height);

    for (unsigned int y = 0; y < map_info_height; y++)
    {
      for (unsigned int x = 0; x < map_info_width; x++)
      {
        unsigned int i = x + (map_info_height - y - 1) * map_info_width;
        if (map_data[i] >= 0 && map_data[i] < 40)
        { // occ [0,0.1)
          fputc(254, out);
        }
        else if (map_data[i] > +50)
        { // occ (0.65,1]
          fputc(000, out);
        }
        else
        { // occ [0.1,0.65]
          fputc(205, out);
        }
      }
    }

    fclose(out);

    std::string mapmetadatafile = save_path_ + ".yaml";
    printf("Writing map occupancy data to %s\n", mapmetadatafile.c_str());
    FILE *yaml = fopen(mapmetadatafile.c_str(), "w");

    /*
      https://blog.csdn.net/jppdss/article/details/131580728
      resolution: 0.100000
      origin: [0.000000, 0.000000, 0.000000]
      #
      negate: 0
      occupied_thresh: 0.65
      free_thresh: 0.196
     */

    fprintf(yaml, "image: %s\n"
                  "resolution: %f\n"
                  "origin: [%f, %f, %f]\n"
                  "negate: 0\n"
                  "occupied_thresh: 0.65\n"
                  "free_thresh: 0.196\n\n",
            mapdatafile.c_str(), resolution_, map_info_origin_position_x, map_info_origin_position_y, offset_yaw_);

    fclose(yaml);
    printf("Done\n");
  }

  void convert_from_pgm()
  {
    std::string mapdatafile = save_path_ + ".pgm";
    FILE *fp = fopen(mapdatafile.c_str(), "rb");
    if (!fp)
    {
      printf("Couldn't read map file from %s\n", mapdatafile.c_str());
      return;
    }

    int line = 0;
    char header[1024] = {0};
    while (line < 2)
    {
      fgets(header, 1024, fp);
      if (header[0] != '#')
      {
        ++line;
      }
    }
    sscanf(header, "%u %u\n", &map_info_width, &map_info_height);
    fgets(header, 20, fp);

    map_data.resize(map_info_width * map_info_height);
    map_data.assign(map_info_width * map_info_height, 0);

    for (unsigned int y = 0; y < map_info_height; y++)
    {
      for (unsigned int x = 0; x < map_info_width; x++)
      {
        unsigned int i = x + (map_info_height - y - 1) * map_info_width;
        auto tmp = fgetc(fp);
        if (tmp == 000)
          map_data[i] = 100;
      }
    }
    fclose(fp);
    printf("Received a %d X %d map @ %.3f m/pix\n",
           map_info_width, map_info_height, resolution_);

    char c;
    std::string format;
    std::string mapmetadatafile = save_path_ + ".yaml";
    printf("Reading map occupancy data from %s\n", mapmetadatafile.c_str());
    std::ifstream file(mapmetadatafile, std::ios::binary);
    file >> format >> format;
    file >> format >> resolution_;
    file >> format >> c >> map_info_origin_position_x >> c >> map_info_origin_position_y >> c >> offset_yaw_ >> c;
    file.close();

    printf("image: %s\n"
           "resolution: %f\n"
           "origin: [%f, %f, %f]\n"
           "negate: 0\n"
           "occupied_thresh: 0.65\n"
           "free_thresh: 0.196\n",
           mapmetadatafile.c_str(), resolution_, map_info_origin_position_x, map_info_origin_position_y, offset_yaw_);
    printf("Done\n");
  }

  double resolution_;
  double offset_yaw_{0};
  std::string save_path_;

  uint32_t map_info_width;
  uint32_t map_info_height;
  double map_info_origin_position_x;
  double map_info_origin_position_y;
  std::vector<int8_t> map_data;
};
