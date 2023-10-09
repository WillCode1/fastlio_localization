#pragma once
#include <iostream>
#include <Eigen/Core>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree.h>
#include <pcl/common/transforms.h>
#include <omp.h>
#include "utility/Header.h"


struct BnbOptions
{
    // common
    double linear_xy_window_size = 2;                     // meter
    double linear_z_window_size = 0.5;                    // meter
    double angular_search_window = DEG2RAD(180);          // radian
    std::vector<double> pc_resolutions = {0.2, 0.3, 0.5}; // meter
    int bnb_depth = 3;
    double min_score = 0.3; // for pruning, speed up
    double enough_score = 0.8;

    // min resolution
    double min_xy_resolution = 0.2;
    double min_z_resolution = 0.125;
    double min_angular_resolution = 0.1;

    int thread_num = 4;
    double filter_size_scan = 0.1;
    bool debug_mode = false;

    // double max_range = DBL_MAX;
    // double elevation_upper_thre = DBL_MAX;
    // double elevation_lower_thre = -DBL_MAX;

    // 3D
    // int full_resolution_depth = 3;
};

class PrecomputationGridStack3D
{
public:
    PrecomputationGridStack3D(PointCloudType::Ptr map, const BnbOptions &match_option)
    {
        assert(match_option.bnb_depth > 0);
        assert(match_option.pc_resolutions.size() >= match_option.bnb_depth);

        precomputation_grids_.reserve(match_option.bnb_depth);
        for (auto i = 0; i < match_option.bnb_depth; ++i)
        {
            precomputation_grids_.emplace_back(match_option.pc_resolutions[i]);
            auto& octree = precomputation_grids_.back();
            octree.setInputCloud(map);
            octree.addPointsFromInputCloud();
        }
    }

    const pcl::octree::OctreePointCloudSearch<PointType> &Get(int depth) const
    {
        return precomputation_grids_.at(depth);
    }

    int max_depth() const { return precomputation_grids_.size() - 1; }

private:
    std::vector<pcl::octree::OctreePointCloudSearch<PointType>> precomputation_grids_;
};

struct Pose
{
    Pose(double x = 0, double y = 0, double z = 0, double roll = 0, double pitch = 0, double yaw = 0)
        : x(x), y(y), z(z), roll(roll), pitch(pitch), yaw(yaw) {}

    Eigen::Matrix4d toMatrix4d() const
    {
        Eigen::Vector3d translation(x, y, z);
        Eigen::Vector3d eulerAngles(roll, pitch, yaw);
        return EigenMath::CreateAffineMatrix(translation, eulerAngles);
    }

    Pose &operator+=(const Eigen::Vector4d &offset)
    {
        x += offset(0);
        y += offset(1);
        z += offset(2);
        yaw += offset(3);
        return *this;
    }

    double x;
    double y;
    double z;
    double roll;
    double pitch;
    double yaw;
};

struct DiscretePose3D
{
    DiscretePose3D(const Pose &initial_pose, const BnbOptions &match_option)
    {
        assert(match_option.linear_xy_window_size > 0);
        assert(match_option.linear_z_window_size > 0);
        assert(match_option.angular_search_window > 0);
        assert(match_option.min_xy_resolution > 0);
        assert(match_option.min_z_resolution > 0);
        assert(match_option.min_angular_resolution > 0);

        double init_discrete_x = initial_pose.x - match_option.linear_xy_window_size;
        double init_discrete_y = initial_pose.y - match_option.linear_xy_window_size;
        double init_discrete_z = initial_pose.z - match_option.linear_z_window_size;
        double init_discrete_angular = initial_pose.yaw - match_option.angular_search_window;

        auto max_depth = 1 << (match_option.bnb_depth - 1);
        candidate_xy_part = (int)std::ceil(2. * match_option.linear_xy_window_size / max_depth / match_option.min_xy_resolution);
        candidate_z_part = (int)std::ceil(2. * match_option.linear_z_window_size / max_depth / match_option.min_z_resolution);
        candidate_angular_part = (int)std::ceil(2. * match_option.angular_search_window / max_depth / match_option.min_angular_resolution);

        discrete_xy_step = 2 * match_option.linear_xy_window_size / candidate_xy_part;
        discrete_z_step = 2 * match_option.linear_z_window_size / candidate_z_part;
        discrete_angular_step = 2 * match_option.angular_search_window / candidate_angular_part;

        for (auto discrete_x = init_discrete_x; discrete_x <= initial_pose.x + match_option.linear_xy_window_size; discrete_x += discrete_xy_step)
        {
            for (auto discrete_y = init_discrete_y; discrete_y <= initial_pose.y + match_option.linear_xy_window_size; discrete_y += discrete_xy_step)
            {
                for (auto discrete_z = init_discrete_z; discrete_z <= initial_pose.z + match_option.linear_z_window_size; discrete_z += discrete_z_step)
                {
                    for (auto discrete_angular = init_discrete_angular; discrete_angular <= initial_pose.yaw + match_option.angular_search_window; discrete_angular += discrete_angular_step)
                    {
                        // discrete_pose.emplace_back(discrete_x, discrete_y, discrete_z, 0, 0, discrete_angular);
                        discrete_pose.emplace_back(discrete_x, discrete_y, discrete_z, initial_pose.roll, initial_pose.pitch, discrete_angular);
                    }
                }
            }
        }
    }

    // 分区数
    int candidate_xy_part;
    int candidate_z_part;
    int candidate_angular_part;
    // 初始分区步长
    double discrete_xy_step;
    double discrete_z_step;
    double discrete_angular_step;
    std::vector<Pose> discrete_pose;
};

struct Candidate3D
{
    Candidate3D(const int discrete_index, const Eigen::Vector4d &offset)
        : discrete_index(discrete_index), offset(offset) {}

    static Candidate3D Unsuccessful()
    {
        return Candidate3D(0, Eigen::Vector4d::Zero());
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Index into the discrete pose vectors.
    int discrete_index;

    // Linear offset from the initial discrete pose in discrete indices.
    Eigen::Vector4d offset;

    float score = -std::numeric_limits<float>::infinity();

    bool operator<(const Candidate3D &other) const { return score < other.score; }
    bool operator>(const Candidate3D &other) const { return score > other.score; }
};

class BranchAndBoundMatcher3D
{
public:
    BranchAndBoundMatcher3D(PointCloudType::Ptr map, const BnbOptions &match_option)
        : thread_num_(match_option.thread_num), precomputation_grid_stack_(map, match_option)
    {
    }

    PointCloudType::Ptr filterScan(const PointCloudType::Ptr &scan, const BnbOptions &match_option)
    {
        PointCloudType::Ptr filter_cloud(new PointCloudType);
        pcl::VoxelGrid<PointType> downSizeFilterSurf;
        downSizeFilterSurf.setLeafSize(match_option.filter_size_scan, match_option.filter_size_scan, match_option.filter_size_scan);
        downSizeFilterSurf.setInputCloud(scan);
        downSizeFilterSurf.filter(*filter_cloud);
        return filter_cloud;
    }

    std::vector<Candidate3D> ComputeLowestResolutionCandidates(const DiscretePose3D& discrete_candidate_pose)
    {
        std::vector<Candidate3D> candidates;
        candidates.reserve(discrete_candidate_pose.discrete_pose.size());
        for (int scan_index = 0; scan_index != discrete_candidate_pose.discrete_pose.size(); ++scan_index)
        {
            candidates.emplace_back(scan_index, Eigen::Vector4d(0, 0, 0, 0));
        }
        return candidates;
    }

    // 计算当前位姿点云在地图中的占据分数
    double calculateOccupancyScore(const int depth, const PointCloudType::Ptr &pointCloud)
    {
        assert(depth <= precomputation_grid_stack_.max_depth());
        double score = 0;
        auto& octree = precomputation_grid_stack_.Get(depth);

        // 查询给定点是否在八叉树网格中
        for (const auto &point : pointCloud->points)
        {
            if (octree.isVoxelOccupiedAtPoint(point))
            {
                ++score;
            }
        }
        return score / pointCloud->points.size();
    }

    void ScoreCandidates(const int depth, const PointCloudType::Ptr &pointCloud,
                         const DiscretePose3D &discrete_candidate_pose, std::vector<Candidate3D> &candidates)
    {
#pragma omp parallel for num_threads(thread_num_)
        // omp mustn't use '!=' / 'range for'
        for (auto i = 0; i < candidates.size(); ++i)
        {
            PointCloudType::Ptr trans_pc(new PointCloudType(pointCloud->points.size(), 1));
            auto candidate_pose = discrete_candidate_pose.discrete_pose[candidates[i].discrete_index];
            candidate_pose += candidates[i].offset;
            const Eigen::Matrix4d &candidate_lidar_pose = candidate_pose.toMatrix4d() * init_lidar_orientation_;
            pcl::transformPointCloud(*pointCloud, *trans_pc, candidate_lidar_pose);
            candidates[i].score = calculateOccupancyScore(depth, trans_pc);
        }
        sort_cnt++;
        std::sort(candidates.begin(), candidates.end(), std::greater<Candidate3D>());
    }

    Candidate3D BranchAndBound(const BnbOptions &match_option, const PointCloudType::Ptr &scan,
                               DiscretePose3D &discrete_candidate_pose, const std::vector<Candidate3D> &candidates,
                               const int candidate_depth, float min_score)
    {
        if (candidate_depth == 0)
        {
            if (match_option.debug_mode)
            {
                printf("bnb_depth = %d, score = %f\n", candidate_depth, candidates.begin()->score);
            }
            // Return the best candidate.
            return *candidates.begin();
        }

        Candidate3D best_high_resolution_candidate = Candidate3D::Unsuccessful();
        best_high_resolution_candidate.score = min_score;
        for (const Candidate3D &candidate : candidates)
        {
            if (candidate.score <= min_score)
            {
                break;
            }
            std::vector<Candidate3D> higher_resolution_candidates;
            const int resolution = 1 << (match_option.bnb_depth - candidate_depth);
            double xy_step = discrete_candidate_pose.discrete_xy_step / resolution;
            double z_step = discrete_candidate_pose.discrete_z_step / resolution;
            double angular_step = discrete_candidate_pose.discrete_angular_step / resolution;
            for (double& yaw : std::vector<double>{0, angular_step})
            {
                if (candidate.offset(3) + yaw > match_option.linear_xy_window_size)
                {
                    break;
                }
                for (double& z : std::vector<double>{0, z_step})
                {
                    if (candidate.offset(2) + z > match_option.linear_xy_window_size)
                    {
                        break;
                    }
                    for (double& y : std::vector<double>{0, xy_step})
                    {
                        if (candidate.offset(1) + y > match_option.linear_xy_window_size)
                        {
                            break;
                        }
                        for (double& x : std::vector<double>{0, xy_step})
                        {
                            if (candidate.offset(0) + x > match_option.linear_xy_window_size)
                            {
                                break;
                            }
                            higher_resolution_candidates.emplace_back(candidate.discrete_index, candidate.offset + Eigen::Vector4d(x, y, z, yaw));
                        }
                    }
                }
            }
            ScoreCandidates(candidate_depth - 1, scan, discrete_candidate_pose, higher_resolution_candidates);
            best_high_resolution_candidate = std::max(
                best_high_resolution_candidate,
                BranchAndBound(match_option, scan, discrete_candidate_pose,
                               higher_resolution_candidates, candidate_depth - 1,
                               best_high_resolution_candidate.score));
            if (match_option.debug_mode)
            {
                printf("bnb_depth = %d, score = %f\n", candidate_depth, best_high_resolution_candidate.score);
            }
        }
        return best_high_resolution_candidate;
    }

    bool MatchWithMatchOptions(const Pose &init_pose, Pose &res_pose,
                               const PointCloudType::Ptr &scan,
                               const BnbOptions &match_option,
                               const Eigen::Matrix4d &lidar_ext,
                               double &score)
    {
        sort_cnt = 0;
        init_lidar_orientation_ = lidar_ext;
        auto filter_scan = filterScan(scan, match_option);
        DiscretePose3D discrete_candidate_pose(init_pose, match_option);
        std::vector<Candidate3D> lowest_resolution_candidates = ComputeLowestResolutionCandidates(discrete_candidate_pose);
        ScoreCandidates(precomputation_grid_stack_.max_depth(), filter_scan, discrete_candidate_pose, lowest_resolution_candidates);

        const Candidate3D best_candidate = BranchAndBound(match_option, filter_scan, discrete_candidate_pose, lowest_resolution_candidates,
                                                          precomputation_grid_stack_.max_depth(), match_option.min_score);

        printf("xy_step = %d, z_step = %d, angular_step = %d, init_candidates_num = %lu, best_score = %f\n",
               discrete_candidate_pose.candidate_xy_part, discrete_candidate_pose.candidate_z_part, discrete_candidate_pose.candidate_angular_part,
               lowest_resolution_candidates.size(), best_candidate.score);
        if (best_candidate.score > match_option.min_score + 1e-6)
        {
            res_pose = discrete_candidate_pose.discrete_pose[best_candidate.discrete_index];
            res_pose += best_candidate.offset;
            score = best_candidate.score;
            return true;
        }
        return false;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int sort_cnt = 0;
private:
    int thread_num_;
    Eigen::Matrix4d init_lidar_orientation_;
    PrecomputationGridStack3D precomputation_grid_stack_;
};

