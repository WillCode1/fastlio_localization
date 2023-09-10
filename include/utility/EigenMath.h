/*
    Created by will on 2022/2/16.
    Eigen中四元数、欧拉角、旋转矩阵、旋转向量
*/
#pragma once
#include <random>
#include <ctime>
#include <iostream>
#include "Eigen/Dense"
using namespace Eigen;

class EigenMath
{
public:
    // 一、旋转向量
    // 1.1 旋转向量转旋转矩阵
    static Eigen::Matrix3d AngleAxis2RotationMatrix(const Eigen::AngleAxisd &angle_axis)
    {
        // 1.0 初始化旋转向量：旋转角为alpha，旋转轴为(x, y, z)
        // Eigen::AngleAxisd rotation_vector(alpha, Vector3d(x, y, z))
        return angle_axis.matrix();
    }

    // 1.2 旋转向量转欧拉角(Z - Y - X，即RPY)
    static Eigen::Vector3d AngleAxis2RPY(const Eigen::AngleAxisd &angle_axis)
    {
        return RotationMatrix2RPY2(angle_axis.matrix());
    }

    // 1.3 旋转向量转四元数
    static Eigen::Quaterniond AngleAxis2Quaternion(const Eigen::AngleAxisd &angle_axis)
    {
        return Eigen::Quaterniond(angle_axis);
    }

    // 二、旋转矩阵
    // 2.1 旋转矩阵转旋转向量
    static Eigen::AngleAxisd RotationMatrix2AngleAxis(const Eigen::Matrix3d &rotation_matrix)
    {
        // 2.0 初始化旋转矩阵
        // Eigen::Matrix3d rotation_matrix;
        // rotation_matrix << x_00, x_01, x_02, x_10, x_11, x_12, x_20, x_21, x_22;
        return Eigen::AngleAxisd().fromRotationMatrix(rotation_matrix);
    }

    // 2.2 旋转矩阵转欧拉角(Z - Y - X，即RPY)
    static Eigen::Vector3d RotationMatrix2RPY(const Eigen::Matrix3d &rotation_matrix)
    {
        return rotation_matrix.eulerAngles(0, 1, 2);
    }

    // 2.3 旋转矩阵转四元数
    static Eigen::Quaterniond RotationMatrix2Quaternion(const Eigen::Matrix3d &rotation_matrix)
    {
        return Eigen::Quaterniond(rotation_matrix);
    }

    // 2.4 fix eigen bug: https://blog.csdn.net/qq_36594547/article/details/119218807
    static Eigen::Vector3d RotationMatrix2RPY2(const Eigen::Matrix3d &rotation)
    {
        Eigen::Vector3d n = rotation.col(0);
        Eigen::Vector3d o = rotation.col(1);
        Eigen::Vector3d a = rotation.col(2);

        Eigen::Vector3d rpy(3);
        const double &y = atan2(n(1), n(0));
        const double &p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
        const double &r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
        rpy(0) = r;
        rpy(1) = p;
        rpy(2) = y;
        return rpy;
    }

    // 三、欧拉角
    // 3.1 欧拉角转旋转向量
    static Eigen::AngleAxisd RPY2AngleAxis(const Eigen::Vector3d &eulerAngles)
    {
        // 3.0 初始化欧拉角(Z - Y - X，即RPY)
        // Eigen::Vector3d eulerAngle(yaw, pitch, roll);
        Eigen::AngleAxisd rollAngle(AngleAxisd(eulerAngles(0), Vector3d::UnitX()));
        Eigen::AngleAxisd pitchAngle(AngleAxisd(eulerAngles(1), Vector3d::UnitY()));
        Eigen::AngleAxisd yawAngle(AngleAxisd(eulerAngles(2), Vector3d::UnitZ()));
        Eigen::AngleAxisd angle_axis;
        angle_axis = yawAngle * pitchAngle * rollAngle;
        return angle_axis;
    }

    // 3.2 欧拉角转旋转矩阵
    static Eigen::Matrix3d RPY2RotationMatrix(const Eigen::Vector3d &eulerAngles)
    {
        Eigen::AngleAxisd rollAngle(AngleAxisd(eulerAngles(0), Vector3d::UnitX()));
        Eigen::AngleAxisd pitchAngle(AngleAxisd(eulerAngles(1), Vector3d::UnitY()));
        Eigen::AngleAxisd yawAngle(AngleAxisd(eulerAngles(2), Vector3d::UnitZ()));
        Eigen::Matrix3d rotation_matrix;
        rotation_matrix = yawAngle * pitchAngle * rollAngle;
        return rotation_matrix;
    }

    // 3.3 欧拉角转四元数
    static Eigen::Quaterniond RPY2Quaternion(const Eigen::Vector3d &eulerAngles)
    {
        Eigen::AngleAxisd rollAngle(AngleAxisd(eulerAngles(0), Vector3d::UnitX()));
        Eigen::AngleAxisd pitchAngle(AngleAxisd(eulerAngles(1), Vector3d::UnitY()));
        Eigen::AngleAxisd yawAngle(AngleAxisd(eulerAngles(2), Vector3d::UnitZ()));
        Eigen::Quaterniond quaternion;
        quaternion = yawAngle * pitchAngle * rollAngle;
        return quaternion;
    }

    // 3.4 欧拉角转四元数快速版
    // It's approximately equal if there's only one direction. Three times as fast.
    static Eigen::Quaterniond RPY2QuaternionFast(const Eigen::Vector3d &rpy)
    {
        double scale = 0.5;
        double w = 1.;
        constexpr double kCutoffAngle = 1e-8; // We linearize below this angle.
        if (rpy.squaredNorm() > kCutoffAngle)
        {
            const double norm = rpy.norm();
            scale = sin(norm / 2.) / norm;
            w = cos(norm / 2.);
        }
        const Eigen::Vector3d quaternion_xyz = scale * rpy;
        return Eigen::Quaterniond(w, quaternion_xyz.x(), quaternion_xyz.y(), quaternion_xyz.z());
    }

    // 四、四元数
    // 4.1 四元数转旋转向量
    static Eigen::AngleAxisd Quaternion2AngleAxis(const Eigen::Quaterniond &quaternion)
    {
        return Eigen::AngleAxisd(quaternion);
    }

    // 4.2 四元数转旋转矩阵
    static Eigen::Matrix3d Quaternion2RotationMatrix(const Eigen::Quaterniond &quaternion)
    {
        return quaternion.matrix();
    }

    // 4.3 四元数转欧拉角(Z - Y - X，即RPY)
    static Eigen::Vector3d Quaternion2RPY(const Eigen::Quaterniond &quaternion)
    {
        return RotationMatrix2RPY2(quaternion.matrix());
    }

    // 5.1 pose -> matrix
    static Eigen::Matrix4d CreateAffineMatrix(const Eigen::Vector3d &translation, const Eigen::Vector3d &eulerAngles)
    {
        Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
        transform.topLeftCorner(3, 3) = EigenMath::RPY2RotationMatrix(eulerAngles);
        transform.topRightCorner(3, 1) = translation;
        return transform;
    }

    static Eigen::Matrix4d CreateAffineMatrix(const double &x, const double &y, const double &z, const double &roll, const double &pitch, const double &yaw)
    {
        return CreateAffineMatrix(Eigen::Vector3d(x, y, z), Eigen::Vector3d(roll, pitch, yaw));
    }

    // 5.2 matrix -> pose
    static void DecomposeAffineMatrix(const Eigen::Matrix4d &affine_mat, Eigen::Vector3d &translation, Eigen::Vector3d &eulerAngles)
    {
        translation = affine_mat.topRightCorner(3, 1);
        eulerAngles = EigenMath::RotationMatrix2RPY2(Eigen::Matrix3d(affine_mat.topLeftCorner(3, 3)));
    }

    static void DecomposeAffineMatrix(const Eigen::Matrix4d &affine_mat, double &x, double &y, double &z, double &roll, double &pitch, double &yaw)
    {
        Eigen::Vector3d eulerAngles, translation;
        DecomposeAffineMatrix(affine_mat, translation, eulerAngles);
        x = translation.x();
        y = translation.y();
        z = translation.z();
        roll = eulerAngles.x();
        pitch = eulerAngles.y();
        yaw = eulerAngles.z();
    }
};
