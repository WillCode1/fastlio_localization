#pragma once
#include <unordered_map>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/gicp.h>
#include "DataDef.h"
#include "use-ikfom.hpp"
#include "utility/Header.h"
#include "utility/manually_correct_loop_closure.h"
#include "global_localization/scancontext/Scancontext.h"


class LoopClosure
{
public:
    LoopClosure(const std::shared_ptr<ScanContext::SCManager> scManager)
    {
        copy_keyframe_pose6d.reset(new pcl::PointCloud<PointXYZIRPYT>());
        kdtree_history_keyframe_pose.reset(new pcl::KdTreeFLANN<PointXYZIRPYT>());

        curKeyframeCloud.reset(new PointCloudType());
        prevKeyframeCloud.reset(new PointCloudType());

        loop_vaild_period["odom"] = std::vector<double>();
        loop_vaild_period["scancontext"] = std::vector<double>();
        sc_manager = scManager;
    }

    bool detect_loop_by_scancontext(int &latest_id, int &closest_id, float &sc_yaw_rad)
    {
        auto detectResult = sc_manager->detectLoopClosureID(50); // first: nn index, second: yaw diff
        closest_id = detectResult.first;
        sc_yaw_rad = detectResult.second;

        if (closest_id == -1)
          return false;

        latest_id = copy_keyframe_pose6d->size() - 1;
        LOG_WARN("detect_loop_by_scancontext, id is %d.", closest_id);
        return true;
    }

    bool detect_loop_by_distance(int &latest_id, int &closest_id, const double &lidar_end_time)
    {
        latest_id = copy_keyframe_pose6d->size() - 1; // 当前关键帧索引
        closest_id = -1;

        // 当前帧已经添加过闭环对应关系，不再继续添加
        auto it = loop_constraint_records.find(latest_id);
        if (it != loop_constraint_records.end())
            return false;

        // 在历史关键帧中查找与当前关键帧距离最近的关键帧
        std::vector<int> indices;
        std::vector<float> distances;
        kdtree_history_keyframe_pose->setInputCloud(copy_keyframe_pose6d);
        kdtree_history_keyframe_pose->radiusSearch(copy_keyframe_pose6d->back(), loop_closure_search_radius, indices, distances, 0);
        for (int i = 0; i < (int)indices.size(); ++i)
        {
            int id = indices[i];
            if (abs(copy_keyframe_pose6d->points[id].time - lidar_end_time) > loop_closure_search_time_interval)
            {
                closest_id = id;
                break;
            }
        }
        if (closest_id == -1 || latest_id == closest_id)
            return false;
        return true;
    }

    /**
     * 提取key索引的关键帧前后相邻若干帧的关键帧特征点集合，降采样
     */
    void loop_find_near_keyframes(PointCloudType::Ptr &near_keyframes, const int &key, const int &search_num,
                                  const deque<PointCloudType::Ptr> &keyframe_scan)
    {
        // 提取key索引的关键帧前后相邻若干帧的关键帧特征点集合
        near_keyframes->clear();
        int cloudSize = copy_keyframe_pose6d->size();
        for (int i = -search_num; i <= search_num; ++i)
        {
            int key_near = key + i;
            if (key_near < 0 || key_near >= cloudSize)
                continue;

            *near_keyframes += *pointcloudLidarToWorld(keyframe_scan[key_near], copy_keyframe_pose6d->points[key_near]);
        }

        if (near_keyframes->empty())
            return;

        icp_downsamp_filter.setLeafSize(icp_downsamp_size, icp_downsamp_size, icp_downsamp_size);
        icp_downsamp_filter.setInputCloud(near_keyframes);
        icp_downsamp_filter.filter(*near_keyframes);
    }

    void run(const double lidar_end_time, const deque<PointCloudType::Ptr> &keyframe_scan)
    {
        if (copy_keyframe_pose6d->points.size() < loop_keyframe_num_thld)
        {
            return;
        }

        const double &dartion_time = copy_keyframe_pose6d->back().time - copy_keyframe_pose6d->front().time;
        // 当前关键帧索引，候选闭环匹配帧索引
        int loop_key_cur;
        int loop_key_ref;
        float sc_yaw_rad = -1; // sc2右移 <=> lidar左转 <=> 左+sc_yaw_rad

        {
            // 1.在历史关键帧中查找与当前关键帧距离最近的关键帧
            if (!is_vaild_loop_time_period(dartion_time, loop_vaild_period["odom"]) || !detect_loop_by_distance(loop_key_cur, loop_key_ref, lidar_end_time))
            {
                return;
            }
            // 2.scan context
            if (is_vaild_loop_time_period(dartion_time, loop_vaild_period["scancontext"]))
                if (!detect_loop_by_scancontext(loop_key_cur, loop_key_ref, sc_yaw_rad))
                    return;
        }

        // extract cloud
        PointCloudType::Ptr cur_keyframe_cloud(new PointCloudType());
        PointCloudType::Ptr ref_near_keyframe_cloud(new PointCloudType());
        {
            loop_find_near_keyframes(cur_keyframe_cloud, loop_key_cur, 0, keyframe_scan);
            loop_find_near_keyframes(ref_near_keyframe_cloud, loop_key_ref, keyframe_search_num, keyframe_scan);
            if (cur_keyframe_cloud->size() < 300 || ref_near_keyframe_cloud->size() < 1000)
            {
                return;
            }

            // publish loop submap
            *prevKeyframeCloud = *ref_near_keyframe_cloud;
        }

        // GICP match
        pcl::GeneralizedIterativeClosestPoint<PointType, PointType> gicp;
        gicp.setMaxCorrespondenceDistance(loop_closure_search_radius * 2);
        gicp.setMaximumIterations(100);
        gicp.setTransformationEpsilon(1e-6);
        gicp.setEuclideanFitnessEpsilon(1e-6);
        gicp.setRANSACIterations(0);

        gicp.setInputSource(cur_keyframe_cloud);
        gicp.setInputTarget(ref_near_keyframe_cloud);
        PointCloudType::Ptr unused_result(new PointCloudType());
        if (sc_yaw_rad >= 0)
        {
            const auto &pose_ref = copy_keyframe_pose6d->points[loop_key_ref];
            Eigen::Matrix4f pose_ref_mat = EigenMath::CreateAffineMatrix(V3D(pose_ref.x, pose_ref.y, pose_ref.z), V3D(pose_ref.roll, pose_ref.pitch, pose_ref.yaw + sc_yaw_rad)).cast<float>();

            const auto &pose_cur = copy_keyframe_pose6d->back();
            Eigen::Matrix4f pose_cur_mat = EigenMath::CreateAffineMatrix(V3D(pose_cur.x, pose_cur.y, pose_cur.z), V3D(pose_cur.roll, pose_cur.pitch, pose_cur.yaw)).cast<float>();
            gicp.align(*unused_result, pose_cur_mat.inverse() * pose_ref_mat);
        }
        else
            gicp.align(*unused_result);

        if (gicp.hasConverged() == false || gicp.getFitnessScore() > loop_closure_fitness_score_thld)
        {
            LOG_ERROR("loop closure failed, %d, %.3f, %.3f", gicp.hasConverged(), gicp.getFitnessScore(), loop_closure_fitness_score_thld);
            return;
        }

        // publish corrected cloud
        {
            PointCloudType::Ptr corrected_cloud(new PointCloudType());
            pcl::transformPointCloud(*cur_keyframe_cloud, *corrected_cloud, gicp.getFinalTransformation());
            *curKeyframeCloud = *corrected_cloud;
        }

        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        correctionLidarFrame = gicp.getFinalTransformation();
        float noiseScore = gicp.getFitnessScore();

        if (manually_fine_tune_loop_closure)
        {
            pcl::getTranslationAndEulerAngles(correctionLidarFrame, trans_state[0], trans_state[1], trans_state[2], trans_state[3], trans_state[4], trans_state[5]);
            noiseScore = manually_adjust_loop_closure(ref_near_keyframe_cloud, cur_keyframe_cloud, correctionLidarFrame);
        }

        // Get current frame wrong pose
        Eigen::Affine3f tWrong = pclPointToAffine3f(copy_keyframe_pose6d->points[loop_key_cur]);
        // Get current frame corrected pose
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;
        pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(x, y, z));
        // Get reference frame pose
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_keyframe_pose6d->points[loop_key_ref]);
        gtsam::Vector Vector6(6);
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        gtsam::noiseModel::Diagonal::shared_ptr constraintNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);

        loop_mtx.lock();
        loop_constraint.loop_indexs.push_back(make_pair(loop_key_cur, loop_key_ref));
        loop_constraint.loop_pose_correct.push_back(poseFrom.between(poseTo));
        loop_constraint.loop_noise.push_back(constraintNoise);
        loop_mtx.unlock();

        LOG_WARN("Loop Factor Added by keyframe id = %d, noise = %.3f.", loop_key_ref, noiseScore);
        loop_constraint_records[loop_key_cur] = loop_key_ref;
    }

    void get_loop_constraint(LoopConstraint &loop_constr)
    {
        loop_mtx.lock();
        loop_constr = loop_constraint;
        loop_constraint.clear();
        loop_mtx.unlock();
    }

    bool is_vaild_loop_time_period(const double &time, const std::vector<double> &vaild_period)
    {
        if (vaild_period.empty())
            return true;
        if (vaild_period.size() % 2 != 0)
        {
            LOG_ERROR("time_period size must be double!");
            return true;
        }

        for (auto i = 0; i < vaild_period.size(); i = i + 2)
        {
            if (vaild_period[i] > vaild_period[i + 1])
            {
                LOG_ERROR("time_period must before early than after!");
                continue;
            }
            if (time >= vaild_period[i] && time <= vaild_period[i + 1])
                return true;
        }

        return false;
    }

public:
    bool manually_fine_tune_loop_closure = false;
    std::unordered_map<std::string, std::vector<double>> loop_vaild_period;
    std::mutex loop_mtx;
    int loop_keyframe_num_thld = 50;
    float loop_closure_search_radius = 10;
    float loop_closure_search_time_interval = 30;
    int keyframe_search_num = 20;
    float loop_closure_fitness_score_thld = 0.05;
    float icp_downsamp_size = 0.1;
    pcl::VoxelGrid<PointType> icp_downsamp_filter;

    pcl::PointCloud<PointXYZIRPYT>::Ptr copy_keyframe_pose6d;
    pcl::KdTreeFLANN<PointXYZIRPYT>::Ptr kdtree_history_keyframe_pose;

    unordered_map<int, int> loop_constraint_records; // <new, old>, keyframe index that has added loop constraint
    LoopConstraint loop_constraint;
    std::shared_ptr<ScanContext::SCManager> sc_manager; // scan context

    // for visualize
    PointCloudType::Ptr curKeyframeCloud;
    PointCloudType::Ptr prevKeyframeCloud;
};
