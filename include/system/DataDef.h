#pragma once
#include <deque>
#include <iomanip>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include "utility/Header.h"
#include "frontend/use-ikfom.hpp"

struct ImuState
{
    ImuState(const double &time = 0, const V3D &a = ZERO3D, const V3D &g = ZERO3D,
             const V3D &v = ZERO3D, const V3D &p = ZERO3D, const M3D &r = EYE3D)
        : offset_time(time), acc(a), gyr(g), vel(v), pos(p), rot(r) {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double offset_time;
    V3D acc;
    V3D gyr;
    V3D vel;
    V3D pos;
    M3D rot;
};

enum FrontendOdometryType
{
    Fastlio,
    Pointlio
};

struct ImuData
{
    using Ptr = std::shared_ptr<ImuData>;

    ImuData(const double &t = 0, const V3D &av = ZERO3D, const V3D &la = ZERO3D)
    {
        timestamp = t;
        angular_velocity = av;
        linear_acceleration = la;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double timestamp;
    V3D angular_velocity;
    V3D linear_acceleration;
};

struct MeasureCollection
{
    MeasureCollection()
    {
        lidar_beg_time = 0.0;
        lidar.reset(new PointCloudType());
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double lidar_beg_time;
    double lidar_end_time;
    PointCloudType::Ptr lidar;
    deque<ImuData::Ptr> imu;
};

struct LoopConstraint
{
    void clear()
    {
        loop_indexs.clear();
        loop_pose_correct.clear();
        loop_noise.clear();
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    vector<pair<int, int>> loop_indexs;
    vector<gtsam::Pose3> loop_pose_correct;
    vector<gtsam::noiseModel::Diagonal::shared_ptr> loop_noise;
};

template <typename ikfom_state>
void pointLidarToWorld(V3D const &p_lidar, V3D &p_imu, const ikfom_state &state)
{
    p_imu = state.offset_R_L_I.normalized() * p_lidar + state.offset_T_L_I;
}

template <typename ikfom_state>
void pointLidarToWorld(PointType const &pi, PointType &po, const ikfom_state &state)
{
    V3D p_lidar(pi.x, pi.y, pi.z);
    V3D p_global(state.rot.normalized() * (state.offset_R_L_I.normalized() * p_lidar + state.offset_T_L_I) + state.pos);

    po.x = p_global(0);
    po.y = p_global(1);
    po.z = p_global(2);
    po.intensity = pi.intensity;
}

inline void pointLidarToWorld(PointType const &pi, PointType &po, const M3D &lidar_rot, const V3D &lidar_pos)
{
    V3D p_lidar(pi.x, pi.y, pi.z);
    V3D p_global(lidar_rot * p_lidar + lidar_pos);

    po.x = p_global(0);
    po.y = p_global(1);
    po.z = p_global(2);
    po.intensity = pi.intensity;
}

template <typename ikfom_state>
void pointcloudLidarToWorld(const PointCloudType::Ptr cloud_in, PointCloudType::Ptr cloud_out, const ikfom_state &state)
{
    auto cloud_num = cloud_in->points.size();
    cloud_out->resize(cloud_num);

    // imu pose -> lidar pose
    M3D lidar_rot = state.rot.toRotationMatrix() * state.offset_R_L_I;
    V3D lidar_pos = state.rot * state.offset_T_L_I + state.pos;

#pragma omp parallel for num_threads(MP_PROC_NUM)
    for (int i = 0; i < cloud_num; i++)
    {
        pointLidarToWorld(cloud_in->points[i], cloud_out->points[i], lidar_rot, lidar_pos);
    }
}

inline PointCloudType::Ptr pointcloudKeyframeToWorld(const PointCloudType::Ptr &cloud_in, const PointXYZIRPYT &pose)
{
    int cloudSize = cloud_in->size();
    PointCloudType::Ptr cloud_out(new PointCloudType(cloudSize, 1));
    cloud_out->resize(cloudSize);

    const M3D &state_rot = EigenMath::RPY2RotationMatrix(V3D(pose.roll, pose.pitch, pose.yaw));
    const V3D &state_pos = V3D(pose.x, pose.y, pose.z);

#pragma omp parallel for num_threads(MP_PROC_NUM)
    for (int i = 0; i < cloudSize; ++i)
    {
        V3D p_lidar(cloud_in->points[i].x, cloud_in->points[i].y, cloud_in->points[i].z);
        V3D p_global(state_rot * p_lidar + state_pos);
        cloud_out->points[i].x = p_global(0);
        cloud_out->points[i].y = p_global(1);
        cloud_out->points[i].z = p_global(2);
        cloud_out->points[i].intensity = cloud_in->points[i].intensity;
    }
    return cloud_out;
}

#define NO_LOGER

class LogAnalysis
{
public:
    LogAnalysis()
    {
        frame_num = preprocess_time = preprocess_avetime = 0;
        imu_process_avetime = downsample_avetime = kdtree_search_avetime = match_avetime = cal_H_avetime = 0;
        meas_update_avetime = kdtree_incremental_avetime = kdtree_delete_avetime = map_incre_avetime = map_remove_avetime = total_avetime = 0;

#ifndef NO_LOGER
        fout_predict = fopen(DEBUG_FILE_DIR("state_predict.txt").c_str(), "w");
        fout_update = fopen(DEBUG_FILE_DIR("state_update.txt").c_str(), "w");
        fout_fastlio_log = fopen(DEBUG_FILE_DIR("fast_lio_log.csv").c_str(), "w");

        if (fout_predict && fout_update)
            cout << "~~~~" << ROOT_DIR << " file opened" << endl;
        else
            cout << "~~~~" << ROOT_DIR << " doesn't exist" << endl;
#endif
    }

    ~LogAnalysis()
    {
        fclose(fout_predict);
        fclose(fout_update);
        fclose(fout_fastlio_log);
    }

    template <typename ikfom_state>
    void print_pose(const ikfom_state &state, const std::string &print)
    {
        const auto &xyz = state.pos;
        const auto &rpy = EigenMath::Quaternion2RPY(state.rot);
        LOG_INFO("%s (xyz, rpy): (%.5f, %.5f, %.5f, %.5f, %.5f, %.5f)", print.c_str(), xyz(0), xyz(1), xyz(2), RAD2DEG(rpy(0)), RAD2DEG(rpy(1)), RAD2DEG(rpy(2)));
    }

    template <typename ikfom_state>
    void print_extrinsic(const ikfom_state &state, bool need_print)
    {
        const auto &offset_xyz = state.offset_T_L_I;
        const auto &offset_rpy = EigenMath::Quaternion2RPY(state.offset_R_L_I);
        LOG_INFO_COND(need_print, "extrinsic_est: (%.5f, %.5f, %.5f, %.5f, %.5f, %.5f)", offset_xyz(0), offset_xyz(1), offset_xyz(2), RAD2DEG(offset_rpy(0)), RAD2DEG(offset_rpy(1)), RAD2DEG(offset_rpy(2)));
    }

    template <typename ikfom_state>
    void dump_state_to_log(FILE *fp, const ikfom_state &state, const double &delta_time)
    {
        if (!runtime_log)
            return;

        V3D rot_ang = EigenMath::Quaternion2RPY(state.rot);
        V3D ext_rot_LI = EigenMath::Quaternion2RPY(state.offset_R_L_I);
        fprintf(fp, "%lf ", delta_time);
        fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                                  // Angle
        fprintf(fp, "%lf %lf %lf ", state.pos(0), state.pos(1), state.pos(2));                            // Pos
        fprintf(fp, "%lf %lf %lf ", state.vel(0), state.vel(1), state.vel(2));                            // Vel
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                       // omega
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                       // Acc
        fprintf(fp, "%lf %lf %lf ", state.bg(0), state.bg(1), state.bg(2));                               // Bias_g
        fprintf(fp, "%lf %lf %lf ", state.ba(0), state.ba(1), state.ba(2));                               // Bias_a
        fprintf(fp, "%lf %lf %lf ", ext_rot_LI(0), ext_rot_LI(1), ext_rot_LI(2));                         // ext_R_LI
        fprintf(fp, "%lf %lf %lf ", state.offset_T_L_I(0), state.offset_T_L_I(1), state.offset_T_L_I(2)); // ext_T_LI
        fprintf(fp, "%lf %lf %lf", state.gravity(0), state.gravity(1), state.gravity(2));                 // gravity
        fprintf(fp, "\n");
        fflush(fp);
    }

    void dump_state_to_log(FILE *fp, const state_ikfom &state, const double &delta_time)
    {
        if (!runtime_log)
            return;

        V3D rot_ang = EigenMath::Quaternion2RPY(state.rot);
        V3D ext_rot_LI = EigenMath::Quaternion2RPY(state.offset_R_L_I);
        fprintf(fp, "%lf ", delta_time);
        fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                                  // Angle
        fprintf(fp, "%lf %lf %lf ", state.pos(0), state.pos(1), state.pos(2));                            // Pos
        fprintf(fp, "%lf %lf %lf ", state.vel(0), state.vel(1), state.vel(2));                            // Vel
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                       // omega
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                       // Acc
        fprintf(fp, "%lf %lf %lf ", state.bg(0), state.bg(1), state.bg(2));                               // Bias_g
        fprintf(fp, "%lf %lf %lf ", state.ba(0), state.ba(1), state.ba(2));                               // Bias_a
        fprintf(fp, "%lf %lf %lf ", ext_rot_LI(0), ext_rot_LI(1), ext_rot_LI(2));                         // ext_R_LI
        fprintf(fp, "%lf %lf %lf ", state.offset_T_L_I(0), state.offset_T_L_I(1), state.offset_T_L_I(2)); // ext_T_LI
        fprintf(fp, "%lf %lf %lf", state.grav[0], state.grav[1], state.grav[2]);                          // gravity
        fprintf(fp, "\n");
        fflush(fp);
    }

    void dump_state_to_log(FILE *fp, const state_output &state, const double &delta_time)
    {
        if (!runtime_log)
            return;

        V3D rot_ang = EigenMath::Quaternion2RPY(state.rot);
        V3D ext_rot_LI = EigenMath::Quaternion2RPY(state.offset_R_L_I);
        fprintf(fp, "%lf ", delta_time);
        fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                                  // Angle
        fprintf(fp, "%lf %lf %lf ", state.pos(0), state.pos(1), state.pos(2));                            // Pos
        fprintf(fp, "%lf %lf %lf ", state.vel(0), state.vel(1), state.vel(2));                            // Vel
        fprintf(fp, "%lf %lf %lf ", state.omg(0), state.omg(1), state.omg(2));                            // omega
        fprintf(fp, "%lf %lf %lf ", state.acc(0), state.acc(1), state.acc(2));                            // Acc
        fprintf(fp, "%lf %lf %lf ", state.bg(0), state.bg(1), state.bg(2));                               // Bias_g
        fprintf(fp, "%lf %lf %lf ", state.ba(0), state.ba(1), state.ba(2));                               // Bias_a
        fprintf(fp, "%lf %lf %lf ", ext_rot_LI(0), ext_rot_LI(1), ext_rot_LI(2));                         // ext_R_LI
        fprintf(fp, "%lf %lf %lf ", state.offset_T_L_I(0), state.offset_T_L_I(1), state.offset_T_L_I(2)); // ext_T_LI
        fprintf(fp, "%lf %lf %lf", state.gravity(0), state.gravity(1), state.gravity(2));                 // gravity
        fprintf(fp, "\n");
        fflush(fp);
    }

    void output_fastlio_log_to_csv(const double &lidar_beg_time)
    {
        if (!runtime_log)
            return;
        static bool first = true;
        if (first)
        {
            fprintf(fout_fastlio_log, "timestamp,total time,feats_undistort size,incremental time,search time,delete size,delete time,"
                                      "kdtree size,kdtree size end,add point size,preprocess time\n");
            first = false;
        }

        fprintf(fout_fastlio_log, "%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",
                lidar_beg_time, total_time, feats_undistort_size, kdtree_incremental_time, kdtree_search_time,
                kdtree_delete_counter, kdtree_delete_time, kdtree_size, kdtree_size_end, add_point_size, preprocess_time);
    }

    void resetTimer()
    {
        timer.restart();

        imu_process_time = 0;
        downsample_time = 0;

        kdtree_search_time = 0;
        match_time = 0;
        cal_H_time = 0;
        meas_update_time = 0;

        kdtree_incremental_time = 0;
        kdtree_delete_time = 0;
        map_incre_time = 0;
        map_remove_time = 0;

        total_time = 0;

        feats_undistort_size = 0;
        feats_down_size = 0;
    }

    void print_fastlio_cost_time()
    {
        total_time = preprocess_time + imu_process_time + downsample_time + meas_update_time + map_incre_time + map_remove_time;

        if (!runtime_log)
            return;

        frame_num++;

        preprocess_avetime = (preprocess_avetime * (frame_num - 1) + preprocess_time) / frame_num;
        imu_process_avetime = (imu_process_avetime * (frame_num - 1) + imu_process_time) / frame_num;
        downsample_avetime = (downsample_avetime * (frame_num - 1) + downsample_time) / frame_num;

        kdtree_search_avetime = (kdtree_search_avetime * (frame_num - 1) + kdtree_search_time) / frame_num;
        match_avetime = (match_avetime * (frame_num - 1) + match_time) / frame_num;
        cal_H_avetime = (cal_H_avetime * (frame_num - 1) + cal_H_time) / frame_num;
        meas_update_avetime = (meas_update_avetime * (frame_num - 1) + meas_update_time) / frame_num;

        kdtree_incremental_avetime = (kdtree_incremental_avetime * (frame_num - 1) + kdtree_incremental_time) / frame_num;
        kdtree_delete_avetime = (kdtree_delete_avetime * (frame_num - 1) + kdtree_delete_time) / frame_num;
        map_incre_avetime = (map_incre_avetime * (frame_num - 1) + map_incre_time) / frame_num;
        map_remove_avetime = (map_remove_avetime * (frame_num - 1) + map_remove_time) / frame_num;

        total_avetime = (total_avetime * (frame_num - 1) + total_time) / frame_num;

#if 0
        printf("[ave_time]: feats_undistort: %d, feats_down: %d, preprocess: %0.3f, imu: %0.3f, downsample: %0.3f, search: %0.3f, match: %0.3f, "
               "meas update: %0.3f, map incre: %0.3f, map remove: %0.3f, ave total: %0.3f\n",
               feats_undistort_size, feats_down_size,
               preprocess_avetime, imu_process_avetime, downsample_avetime, kdtree_search_avetime, match_avetime,
               meas_update_avetime, map_incre_avetime, map_remove_avetime, total_avetime);
#else
        printf("[cur_time]: feats_undistort: %d, feats_down: %d, preprocess: %0.3f, imu: %0.3f, downsample: %0.3f, search: %0.3f, match: %0.3f, "
               "meas update: %0.3f, map incre: %0.3f, map remove: %0.3f, total: %0.3f\n",
               feats_undistort_size, feats_down_size,
               preprocess_time, imu_process_time, downsample_time, kdtree_search_time, match_time,
               meas_update_time, map_incre_time, map_remove_time, total_time);
#endif
    }

    static void save_gps_pose(FILE *fp, const V3D &pos, const V3D &eular, const double &time)
    {
        fprintf(fp, "%0.4lf %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f\n", time, pos.x(), pos.y(), pos.z(), eular.x(), eular.y(), eular.z());
    }

    // 0 : not, 1 : TUM
    static void save_trajectory(FILE *fp, const V3D &pos, const QD &quat, const double &time, int save_traj_fmt = 1)
    {
        if (save_traj_fmt == 1)
        {
            fprintf(fp, "%0.4lf %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f\n", time,
                    pos.x(), pos.y(), pos.z(), quat.x(), quat.y(), quat.z(), quat.w());
        }

        fflush(fp);
    }

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int runtime_log = 0;
    bool inited_first_lidar_beg_time = false;
    double first_lidar_beg_time;

    FILE *fout_fastlio_log;
    FILE *fout_predict, *fout_update;
    Timer timer;

    long unsigned int frame_num;
    double preprocess_time, imu_process_time, downsample_time, kdtree_search_time, match_time, cal_H_time;
    double meas_update_time, kdtree_incremental_time, kdtree_delete_time, map_incre_time, map_remove_time, total_time;

    double preprocess_avetime, imu_process_avetime, downsample_avetime, kdtree_search_avetime, match_avetime, cal_H_avetime;
    double meas_update_avetime, kdtree_incremental_avetime, kdtree_delete_avetime, map_incre_avetime, map_remove_avetime, total_avetime;

    int feats_undistort_size = 0, feats_down_size = 0, kdtree_size = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
};
