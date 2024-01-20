#pragma once
#include <Eigen/Dense>

class evo_tool
{
public:
    evo_tool(const std::string& trajectory_path)
    {
        pose_trajectory = fopen(trajectory_path.c_str(), "w");
        fprintf(pose_trajectory, "# target trajectory\n# timestamp tx ty tz qx qy qz qw\n");
        fflush(pose_trajectory);
    }
    ~evo_tool()
    {
        fclose(pose_trajectory);
    }

    /**
     * @brief transform frame_a to frame_b
     * @param extR rot from frame_b to frame_a
     * @param extP pos from frame_b to frame_a
     */
    template <typename T>
    void poseTransformFrame(const Eigen::Quaternion<T> &rot_from, const Eigen::Matrix<T, 3, 1> &pos_from,
                            const Eigen::Quaternion<T> &extR, const Eigen::Matrix<T, 3, 1> &extP,
                            Eigen::Quaternion<T> &rot_to, Eigen::Matrix<T, 3, 1> &pos_to)
    {
        rot_to = rot_from * extR;
        pos_to = rot_from * extP + pos_from;
    }

    /**
     * @brief transform frame_a to frame_b
     * @param extR rot from frame_a to frame_b
     * @param extP pos from frame_a to frame_b
     */
    template <typename T>
    void poseTransformFrame2(const Eigen::Quaternion<T> &rot_from, const Eigen::Matrix<T, 3, 1> &pos_from,
                             const Eigen::Quaternion<T> &extR, const Eigen::Matrix<T, 3, 1> &extP,
                             Eigen::Quaternion<T> &rot_to, Eigen::Matrix<T, 3, 1> &pos_to)
    {
        rot_to = rot_from * extR.conjugate();
        pos_to = pos_from - rot_to * extP;
    }

    void save_trajectory(const Eigen::Vector3d &pos, const Eigen::Quaterniond &quat, const double &time)
    {
        fprintf(pose_trajectory, "%0.4lf %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f\n", time,
                pos.x(), pos.y(), pos.z(), quat.x(), quat.y(), quat.z(), quat.w());
        fflush(pose_trajectory);
    }

    FILE *pose_trajectory;
};
