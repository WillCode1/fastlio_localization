#pragma once
#include <math.h>
#include <Eigen/Core>

#define SKEW_SYM_MATRX(v) 0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0

/****** SO3 math ******/
// Hat (skew) operator
template <typename T>
Eigen::Matrix<T, 3, 3> hat(const Eigen::Matrix<T, 3, 1> &v)
{
    Eigen::Matrix<T, 3, 3> skew_sym_mat;
    skew_sym_mat << 0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0;
    return skew_sym_mat;
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3> hat(const Eigen::MatrixBase<Derived> &q)
{
    Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
    ans << typename Derived::Scalar(0), -q(2), q(1),
        q(2), typename Derived::Scalar(0), -q(0),
        -q(1), q(0), typename Derived::Scalar(0);
    return ans;
}

template <typename T>
Eigen::Matrix<T, 3, 3> Exp(const Eigen::Matrix<T, 3, 1> &ang)
{
    T ang_norm = ang.norm();
    Eigen::Matrix<T, 3, 3> Eye3 = Eigen::Matrix<T, 3, 3>::Identity();
    if (ang_norm > 1e-7)
    {
        Eigen::Matrix<T, 3, 1> r_axis = ang / ang_norm;
        Eigen::Matrix<T, 3, 3> K;
        K << SKEW_SYM_MATRX(r_axis);
        /// Roderigous Tranformation
        return Eye3 + std::sin(ang_norm) * K + (1.0 - std::cos(ang_norm)) * K * K;
    }
    else
    {
        return Eye3;
    }
}

template <typename T>
Eigen::Matrix<T, 3, 3> Exp(const Eigen::Matrix<T, 3, 1> &ang, const double &dt)
{
    const Eigen::Matrix<T, 3, 1> tmp = ang * dt;
    return Exp(tmp);
}

/* Logrithm of a Rotation Matrix */
template <typename T>
Eigen::Matrix<T, 3, 1> Log(const Eigen::Matrix<T, 3, 3> &R)
{
    T theta = (R.trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (R.trace() - 1));
    Eigen::Matrix<T, 3, 1> K(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
    return (std::abs(theta) < 0.001) ? (0.5 * K) : (0.5 * theta / std::sin(theta) * K);
}

template <typename T>
static Eigen::Matrix<T, 3, 3> J_l(const Eigen::Matrix<T, 3, 1> &r)
{
    Eigen::Matrix<T, 3, 3> A;
    double theta = r.norm();
    Eigen::Matrix<T, 3, 1> ang = r / theta;
    Eigen::Matrix<T, 3, 3> skew = hat(ang);
    if (theta > 1e-11)
    {
        A = sin(theta) / theta * Eigen::Matrix<T, 3, 3>::Identity() - (1 - cos(theta)) / theta * skew + (1 - sin(theta) / theta) * r * r.transpose();
        // A = Eigen::Matrix<T, 3, 3>::Identity() - (1 - cos(theta)) / theta * skew + (1 - sin(theta) / theta) * skew * skew;
    }
    else
        A = Eigen::Matrix<T, 3, 3>::Identity();

    return A;
}

template <typename T>
static Eigen::Matrix<T, 3, 3> J_r(const Eigen::Matrix<T, 3, 1> &r)
{
    return J_l(-r);
}

template <typename T>
static Eigen::Matrix<T, 3, 3> J_l_inv(const Eigen::Matrix<T, 3, 1> &r)
{
    Eigen::Matrix<T, 3, 3> A_inv;
    double half_theta = r.norm() * 0.5;
    Eigen::Matrix<T, 3, 3> skew = hat(r);
    double half_theta_cot_half_theta = half_theta * cos(half_theta) / sin(half_theta);

    if (r.norm() > 1e-11)
    {
        A_inv = half_theta_cot_half_theta * Eigen::Matrix<T, 3, 3>::Identity() - 0.5 * skew + (1 - half_theta_cot_half_theta) * r * r.transpose();
    }
    else
        A_inv = Eigen::Matrix<T, 3, 3>::Identity();
    return A_inv;
}
/****** SO3 math ******/

/****** Quaternion math ******/
// Convert from quaternion to rotation vector
template <typename T>
inline Eigen::Matrix<T, 3, 1> quaternionToRotationVector(const Eigen::Quaternion<T> &qua)
{
    Eigen::Matrix<T, 3, 3> mat = qua.toRotationMatrix();
    Eigen::Matrix<T, 3, 1> rotation_vec;
    Eigen::AngleAxis<T> angle_axis;
    angle_axis.fromRotationMatrix(mat);
    rotation_vec = angle_axis.angle() * angle_axis.axis();
    return rotation_vec;
}

// Calculate the Jacobian with respect to the quaternion
template <typename T>
inline Eigen::Matrix<T, 3, 4> quaternionJacobian(const Eigen::Quaternion<T> &qua, const Eigen::Matrix<T, 3, 1> &vec)
{
    Eigen::Matrix<T, 3, 4> mat;
    Eigen::Matrix<T, 3, 1> quaternion_imaginary(qua.x(), qua.y(), qua.z());

    mat.template block<3, 1>(0, 0) = qua.w() * vec + quaternion_imaginary.cross(vec);
    mat.template block<3, 3>(0, 1) = quaternion_imaginary.dot(vec) * Eigen::Matrix<T, 3, 3>::Identity() + quaternion_imaginary * vec.transpose() - vec * quaternion_imaginary.transpose() - qua.w() * hat(vec);
    return T(2) * mat;
}

// Calculate the Jacobian with respect to the inverse quaternion
template <typename T>
inline Eigen::Matrix<T, 3, 4> quaternionInvJacobian(const Eigen::Quaternion<T> &qua, const Eigen::Matrix<T, 3, 1> &vec)
{
    Eigen::Matrix<T, 3, 4> mat;
    Eigen::Matrix<T, 3, 1> quaternion_imaginary(qua.x(), qua.y(), qua.z());

    mat.template block<3, 1>(0, 0) = qua.w() * vec + vec.cross(quaternion_imaginary);
    mat.template block<3, 3>(0, 1) = quaternion_imaginary.dot(vec) * Eigen::Matrix<T, 3, 3>::Identity() + quaternion_imaginary * vec.transpose() - vec * quaternion_imaginary.transpose() + qua.w() * hat(vec);
    return T(2) * mat;
}

// Calculate the Jacobian rotation vector to quaternion
template <typename T>
inline Eigen::Matrix<T, 3, 4> JacobianRotationVector2Quaternion(const Eigen::Quaternion<T> &qua)
{
    Eigen::Matrix<T, 3, 4> mat;

    T c = 1 / (1 - qua.w() * qua.w());
    T d = acos(qua.w()) / sqrt(1 - qua.w() * qua.w());

    mat.template block<3, 1>(0, 0) = Eigen::Matrix<T, 3, 1>(c * qua.x() * (d * qua.x() - 1),
                                                            c * qua.y() * (d * qua.x() - 1),
                                                            c * qua.z() * (d * qua.x() - 1));
    mat.template block<3, 3>(0, 1) = d * Eigen::Matrix<T, 3, 4>::Identity();
    return T(2) * mat;
}

// https://zhuanlan.zhihu.com/p/67872858
template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qleft(const Eigen::QuaternionBase<Derived> &q)
{
    Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = q.w();
    ans.template block<1, 3>(0, 1) = -q.vec().transpose();
    ans.template block<3, 1>(1, 0) = q.vec();
    ans.template block<3, 3>(1, 1) = q.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() + hat(q.vec());
    return ans;
}

template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qright(const Eigen::QuaternionBase<Derived> &p)
{
    Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = p.w();
    ans.template block<1, 3>(0, 1) = -p.vec().transpose();
    ans.template block<3, 1>(1, 0) = p.vec();
    ans.template block<3, 3>(1, 1) = p.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() - hat(p.vec());
    return ans;
}

// get quaternion from rotation vector
template <typename Derived>
Eigen::Quaternion<typename Derived::Scalar> deltaQ(const Eigen::MatrixBase<Derived> &theta)
{
    typedef typename Derived::Scalar Scalar_t;

    Eigen::Quaternion<Scalar_t> dq;
    Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
    half_theta /= static_cast<Scalar_t>(2.0);
    dq.w() = static_cast<Scalar_t>(1.0);
    dq.x() = half_theta.x();
    dq.y() = half_theta.y();
    dq.z() = half_theta.z();
    return dq;
}
/****** Quaternion math ******/

template <typename T>
Eigen::Matrix<T, 3, 1> RotMtoEuler(const Eigen::Matrix<T, 3, 3> &rot)
{
    T sy = sqrt(rot(0, 0) * rot(0, 0) + rot(1, 0) * rot(1, 0));
    bool singular = sy < 1e-6;
    T x, y, z;
    if (!singular)
    {
        x = atan2(rot(2, 1), rot(2, 2));
        y = atan2(-rot(2, 0), sy);
        z = atan2(rot(1, 0), rot(0, 0));
    }
    else
    {
        x = atan2(-rot(1, 2), rot(1, 1));
        y = atan2(-rot(2, 0), sy);
        z = 0;
    }
    Eigen::Matrix<T, 3, 1> ang(x, y, z);
    return ang;
}
