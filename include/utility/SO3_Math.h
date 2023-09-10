#ifndef SO3_MATH_H
#define SO3_MATH_H

#include <math.h>
#include <Eigen/Core>

class SO3Math{
public:
    static Eigen::Matrix3d get_skew_symmetric(const Eigen::Vector3d &v)
    {
        Eigen::Matrix3d m;
        m << 0, -v(2), v(1),
            v(2), 0, -v(0),
            -v(1), v(0), 0;
        return m;
    }

    static Eigen::Matrix3d Exp(const Eigen::Vector3d &r)
    {
        Eigen::Matrix3d expr;
        double theta = r.norm();
        if (theta < 1e-7)
        {
            expr = Eigen::Matrix3d::Identity();
        }
        else
        {
            Eigen::Matrix3d skew = get_skew_symmetric(r / theta);
            expr = Eigen::Matrix3d::Identity() + sin(theta) * skew + (1 - cos(theta)) * skew * skew;
        }
        return expr;
    }

    static Eigen::Vector3d Log(const Eigen::Matrix3d &R)
    {
        double theta = (R.trace() > 3 - 1e-6) ? 0 : acos((R.trace() - 1) / 2);
        Eigen::Vector3d r(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
        return fabs(theta) < 0.001 ? (0.5 * r) : (0.5 * theta / sin(theta) * r);
    }

    static Eigen::Matrix3d J_l(const Eigen::Vector3d &r)
    {
        Eigen::Matrix3d A;
        double theta = r.norm();
        Eigen::Vector3d ang = r / theta;
        Eigen::Matrix3d skew = get_skew_symmetric(ang);
        if (theta > 1e-11)
        {
            A = Eigen::Matrix3d::Identity() - (1 - cos(theta)) / theta * skew + (1 - sin(theta) / theta) * skew * skew;
        }
        else
            A = Eigen::Matrix3d::Identity();

        return A;
    }

    static Eigen::Matrix3d J_l_inv(const Eigen::Vector3d &r)
    {
        Eigen::Matrix3d A_inv;
        double theta = r.norm();
        double half_theta = theta * 0.5;
        Eigen::Matrix3d skew = get_skew_symmetric(r);

        if (theta > 1e-11)
        {
            A_inv = Eigen::Matrix3d::Identity() - 0.5 * skew + (1 - half_theta * cos(half_theta) / sin(half_theta)) * skew * skew / theta / theta;
        }
        else
            A_inv = Eigen::Matrix3d::Identity();
        return A_inv;
    }
};

#endif
