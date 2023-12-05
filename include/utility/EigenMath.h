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

namespace EigenMath
{
    // 一、旋转向量
    // 1.1 旋转向量转旋转矩阵
    template <typename T>
    static Eigen::Matrix<T, 3, 3> AngleAxis2RotationMatrix(const Eigen::AngleAxis<T> &angle_axis)
    {
        return angle_axis.toRotationMatrix();
    }

    // 1.2 旋转向量转欧拉角(Z - Y - X，即RPY)
    template <typename T>
    static Eigen::Matrix<T, 3, 1> AngleAxis2RPY(const Eigen::AngleAxis<T> &angle_axis)
    {
        return RotationMatrix2RPY(AngleAxis2RotationMatrix(angle_axis));
    }

    // 1.3 旋转向量转四元数
    template <typename T>
    static Eigen::Quaternion<T> AngleAxis2Quaternion(const Eigen::AngleAxis<T> &angle_axis)
    {
        return Eigen::Quaternion<T>(angle_axis);
    }

    // 二、旋转矩阵
    // 2.1 旋转矩阵转旋转向量
    template <typename T>
    static Eigen::AngleAxis<T> RotationMatrix2AngleAxis(const Eigen::Matrix<T, 3, 3> &rotation_matrix)
    {
        return Eigen::AngleAxis<T>().fromRotationMatrix(rotation_matrix);
    }

    // 2.2 旋转矩阵转欧拉角(Z - Y - X，即RPY)
    template <typename T>
    static Eigen::Matrix<T, 3, 1> RotationMatrix2RPY(const Eigen::Matrix<T, 3, 3> &rotation)
    {
        // return rotation_matrix.eulerAngles(0, 1, 2);

        // fix eigen bug: https://blog.csdn.net/qq_36594547/article/details/119218807
        const Eigen::Matrix<T, 3, 1> &n = rotation.col(0);
        const Eigen::Matrix<T, 3, 1> &o = rotation.col(1);
        const Eigen::Matrix<T, 3, 1> &a = rotation.col(2);

        Eigen::Matrix<T, 3, 1> rpy(3);
        const double &y = atan2(n(1), n(0));
        const double &p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
        const double &r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
        rpy(0) = r;
        rpy(1) = p;
        rpy(2) = y;
        return rpy;
    }

    // 2.3 旋转矩阵转四元数
    template <typename T>
    static Eigen::Quaternion<T> RotationMatrix2Quaternion(const Eigen::Matrix<T, 3, 3> &rotation_matrix)
    {
        return Eigen::Quaternion<T>(rotation_matrix);
    }

    // 三、欧拉角
    // 3.1 欧拉角转旋转向量
    template <typename T>
    static Eigen::AngleAxis<T> RPY2AngleAxis(const Eigen::Matrix<T, 3, 1> &eulerAngles)
    {
        Eigen::AngleAxis<T> rollAngle(AngleAxis<T>(eulerAngles(0), Matrix<T, 3, 1>::UnitX()));
        Eigen::AngleAxis<T> pitchAngle(AngleAxis<T>(eulerAngles(1), Matrix<T, 3, 1>::UnitY()));
        Eigen::AngleAxis<T> yawAngle(AngleAxis<T>(eulerAngles(2), Matrix<T, 3, 1>::UnitZ()));
        Eigen::AngleAxis<T> angle_axis;
        angle_axis = yawAngle * pitchAngle * rollAngle;
        return angle_axis;
    }

    // 3.2 欧拉角转旋转矩阵
    template <typename T>
    static Eigen::Matrix<T, 3, 3> RPY2RotationMatrix(const Eigen::Matrix<T, 3, 1> &eulerAngles)
    {
        Eigen::AngleAxis<T> rollAngle(AngleAxis<T>(eulerAngles(0), Matrix<T, 3, 1>::UnitX()));
        Eigen::AngleAxis<T> pitchAngle(AngleAxis<T>(eulerAngles(1), Matrix<T, 3, 1>::UnitY()));
        Eigen::AngleAxis<T> yawAngle(AngleAxis<T>(eulerAngles(2), Matrix<T, 3, 1>::UnitZ()));
        Eigen::Matrix<T, 3, 3> rotation_matrix;
        rotation_matrix = yawAngle * pitchAngle * rollAngle;
        return rotation_matrix;
    }

    // 3.3 欧拉角转四元数
    template <typename T>
    static Eigen::Quaternion<T> RPY2Quaternion(const Eigen::Matrix<T, 3, 1> &eulerAngles)
    {
        Eigen::AngleAxis<T> rollAngle(AngleAxis<T>(eulerAngles(0), Matrix<T, 3, 1>::UnitX()));
        Eigen::AngleAxis<T> pitchAngle(AngleAxis<T>(eulerAngles(1), Matrix<T, 3, 1>::UnitY()));
        Eigen::AngleAxis<T> yawAngle(AngleAxis<T>(eulerAngles(2), Matrix<T, 3, 1>::UnitZ()));
        Eigen::Quaternion<T> quaternion;
        quaternion = yawAngle * pitchAngle * rollAngle;
        return quaternion;
    }

    // 3.4 欧拉角转四元数快速版
    // It's approximately equal if there's only one direction. Three times as fast.
    template <typename T>
    static Eigen::Quaternion<T> RPY2QuaternionFast(const Eigen::Matrix<T, 3, 1> &rpy)
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
        const Eigen::Matrix<T, 3, 1> quaternion_xyz = scale * rpy;
        return Eigen::Quaternion<T>(w, quaternion_xyz.x(), quaternion_xyz.y(), quaternion_xyz.z());
    }

    // 四、四元数
    // 4.1 四元数转旋转向量
    template <typename T>
    static Eigen::AngleAxis<T> Quaternion2AngleAxis(const Eigen::Quaternion<T> &quaternion)
    {
        return Eigen::AngleAxis<T>(quaternion);
    }

    // 4.2 四元数转旋转矩阵
    template <typename T>
    static Eigen::Matrix<T, 3, 3> Quaternion2RotationMatrix(const Eigen::Quaternion<T> &q)
    {
        // return quaternion.normalized().toRotationMatrix();

        // fix eigen bug: The quaternion is required to be normalized, otherwise the result is undefined.
        Eigen::Matrix<T, 3, 3> res;
        const T &nn = q.norm() * q.norm(); // there is fixed!
        const T &tx = T(2) * q.x();
        const T &ty = T(2) * q.y();
        const T &tz = T(2) * q.z();
        const T &twx = tx * q.w();
        const T &twy = ty * q.w();
        const T &twz = tz * q.w();
        const T &txx = tx * q.x();
        const T &txy = ty * q.x();
        const T &txz = tz * q.x();
        const T &tyy = ty * q.y();
        const T &tyz = tz * q.y();
        const T &tzz = tz * q.z();

        res.block(0, 0, 1, 3) << nn - (tyy + tzz), txy - twz, txz + twy;
        res.block(1, 0, 1, 3) << txy + twz, nn - (txx + tzz), tyz - twx;
        res.block(2, 0, 1, 3) << txz - twy, tyz + twx, nn - (txx + tyy);
        return res;
    }

    // 4.3 四元数转欧拉角(Z - Y - X，即RPY)
    template <typename T>
    static Eigen::Matrix<T, 3, 1> Quaternion2RPY(const Eigen::Quaternion<T> &quaternion)
    {
        return RotationMatrix2RPY(Quaternion2RotationMatrix(quaternion));
    }

    // 5.1 pose -> matrix
    template <typename T>
    static Eigen::Matrix<T, 4, 4> CreateAffineMatrix(const Eigen::Matrix<T, 3, 1> &translation, const Eigen::Matrix<T, 3, 1> &eulerAngles)
    {
        Eigen::Matrix<T, 4, 4> transform = Eigen::Matrix<T, 4, 4>::Identity();
        transform.topLeftCorner(3, 3) = RPY2RotationMatrix(eulerAngles);
        transform.topRightCorner(3, 1) = translation;
        return transform;
    }

    template <typename T>
    static Eigen::Matrix<T, 4, 4> CreateAffineMatrix(const double &x, const double &y, const double &z, const double &roll, const double &pitch, const double &yaw)
    {
        return CreateAffineMatrix(Eigen::Matrix<T, 3, 1>(x, y, z), Eigen::Matrix<T, 3, 1>(roll, pitch, yaw));
    }

    // 5.2 matrix -> pose
    template <typename T>
    static void DecomposeAffineMatrix(const Eigen::Matrix<T, 4, 4> &affine_mat, Eigen::Matrix<T, 3, 1> &translation, Eigen::Matrix<T, 3, 1> &eulerAngles)
    {
        translation = affine_mat.topRightCorner(3, 1);
        eulerAngles = RotationMatrix2RPY(Eigen::Matrix<T, 3, 3>(affine_mat.topLeftCorner(3, 3)));
    }

    template <typename T>
    static void DecomposeAffineMatrix(const Eigen::Matrix<T, 4, 4> &affine_mat, double &x, double &y, double &z, double &roll, double &pitch, double &yaw)
    {
        Eigen::Matrix<T, 3, 1> eulerAngles, translation;
        DecomposeAffineMatrix(affine_mat, translation, eulerAngles);
        x = translation.x();
        y = translation.y();
        z = translation.z();
        roll = eulerAngles.x();
        pitch = eulerAngles.y();
        yaw = eulerAngles.z();
    }
}
