#pragma once
#include <iostream>
#include <cmath>
#include <Eigen/Dense>
using namespace std;

namespace zlam {

// ellipsoid para
constexpr double WGS84_RE = 6378137.0;
constexpr double WGS84_F = (1.0 / 298.257223563);

constexpr double gl_wie = 7.2921151467e-5;
constexpr double gl_meru = gl_wie / 1000;
constexpr double gl_g0 = 9.7803267714;
constexpr double gl_mg = 1e-3 * gl_g0;
constexpr double gl_ug = 1e-3 * gl_mg;
constexpr double gl_mGal = 1e-3 * 0.01;  // milli Gal = 1cm/s^2 ~= 1.0E-6*g0
constexpr double gl_ugpg2 = gl_ug / gl_g0 / gl_g0;
constexpr double gl_ppm = 1e-6;
constexpr double gl_deg = M_PI / 180;   // arcdeg
constexpr double gl_min = gl_deg / 60;  // arcmin
constexpr double gl_sec = gl_min / 60;  // arcsec
constexpr double gl_hur = 3600;
constexpr double gl_dps = gl_deg / 1;              // arcdeg / second
constexpr double gl_dph = gl_deg / gl_hur;         // arcdeg / hour
constexpr double gl_dpss = gl_deg / sqrt(1.0);     // arcdeg / sqrt(second)
constexpr double gl_dpsh = gl_deg / sqrt(gl_hur);  // arcdeg / sqrt(hour)
constexpr double gl_dphpsh = gl_dph / sqrt(gl_hur);
constexpr double gl_Hz = 1;  // Hertz
constexpr double gl_dphpsHz = gl_dph / gl_Hz;
constexpr double gl_mgpsHz = gl_mg / sqrt(gl_Hz);
constexpr double gl_ugpsHz = gl_ug / sqrt(gl_Hz);
constexpr double gl_ugpsh = gl_ug / sqrt(gl_hur);  // ug / sqrt(hour)
constexpr double gl_mpsh = 1 / sqrt(gl_hur);
constexpr double gl_ppmpsh = gl_ppm / sqrt(gl_hur);
constexpr double gl_mil = 2 * M_PI / 6000;
constexpr double gl_nm = 1853;            // nautical mile
constexpr double gl_kn = gl_nm / gl_hur;  // 海里每小时
// added
constexpr double gl_mps = 1;
constexpr double gl_km = 1000;
constexpr double gl_kmph = gl_km / gl_hur;
constexpr double gl_mpr = 1 / WGS84_RE;
constexpr double gl_m = 1.0;
constexpr double gl_cm = gl_m / 100;   // cm
constexpr double gl_cmps = gl_cm / 1;  // cm /s

constexpr double operator"" _deg(long double x) { return x / 180 * M_PI; }
constexpr double operator"" _m(long double x) { return x * gl_m; }
constexpr double operator"" _mps(long double x) { return x * gl_mps; }
constexpr double operator"" _kmph(long double x) { return x * gl_kmph; }
constexpr double operator"" _km(long double x) { return x * gl_km; }
constexpr double operator"" _mpr(long double x) { return x * gl_mpr; }
constexpr double operator"" _mg(long double x) { return x * gl_mg; }
constexpr double operator"" _dps(long double x) { return x * gl_dps; }
constexpr double operator"" _ugpsHz(long double x) { return x * gl_ugpsh; }
constexpr double operator"" _dpss(long double x) { return x * gl_dpss; }
constexpr double operator"" _pixel(long double x) { return x; }
// 时间相关
constexpr double gl_SEC = 1;      // seconds
constexpr double gl_MSEC = 1e-3;  // miliseconds
constexpr double gl_USEC = 1e-6;  // microseconds
constexpr double gl_NSEC = 1e-9;  // Nanosecond

constexpr double operator"" _SEC(long double x) { return x; }             // 秒
constexpr double operator"" _MSEC(long double x) { return x * gl_MSEC; }  // 毫秒
constexpr double operator"" _USEC(long double x) { return x * gl_USEC; }  // 微妙
constexpr double operator"" _MIN(long double x) { return x * 60; }        // 分钟
constexpr double operator"" _HOU(long double x) { return x * 3600; }      // 小时

constexpr double operator""_hz(long double x) { return x; }

class Earth {
private:
    Earth() = delete;

public:
    static void LLH2ECEF(const double *pos, double *xyz, bool is_deg = false) {
        double b = pos[0] * (is_deg ? gl_deg : 1);
        double l = pos[1] * (is_deg ? gl_deg : 1);
        double h = pos[2];
        double n = N(b);
        double cb = cos(b);
        xyz[0] = (n + h) * cb * cos(l);
        xyz[1] = (n + h) * cb * sin(l);
        xyz[2] = (n * (1 - _e1 * _e1) + h) * sin(b);
    }

    static Eigen::Vector3d LLH2ECEF(const Eigen::Vector3d &pos, bool is_deg = false) {
        Eigen::Vector3d xyz = {0, 0, 0};
        LLH2ECEF(pos.data(), xyz.data(), is_deg);
        return xyz;
    }

    static void ECEF2LLH(const Eigen::Vector3d &xyz, Eigen::Vector3d *pos, bool to_deg = false) {
        const double e1_2 = _e1 * _e1;
        double r2 = xyz.head(2).squaredNorm();
        double v = _a;
        double z = xyz[2];
        double sinp = 0;
        for (double zk = 0; fabs(z - zk) >= 1e-4;)
        {
            zk = z;
            sinp = z / sqrt(r2 + z * z);
            v = _a / sqrt(1 - e1_2 * sinp * sinp);
            z = xyz[2] + v * e1_2 * sinp;
        }
        auto temp_x = xyz[2] > 0.0 ? M_PI / 2.0 : -M_PI / 2.0;
        (*pos)[0] = r2 > 1E-12 ? atan(z / sqrt(r2)) : temp_x;
        (*pos)[1] = r2 > 1E-12 ? atan2(xyz[1], xyz[0]) : 0.0;
        (*pos)[2] = sqrt(r2 + z * z) - v;
        if (to_deg)
        {
            pos->head(2) /= gl_deg;
        }
        // std::cout << "void ECEF2LLH(const Vector3d &r_, Vector3d *pos, bool
        // to_deg = false)    " << pos->transpose() << std::endl;
    }

    static Eigen::Vector3d ECEF2LLH(const Eigen::Vector3d &xyz, bool to_deg = false) {
        Eigen::Vector3d pos = Eigen::Vector3d::Zero();
        ECEF2LLH(xyz, &pos, to_deg);
        return pos;
    }

    static Eigen::Matrix3d Pos2Cne(const Eigen::Vector3d &pos, bool is_deg = false) {
        using namespace Eigen;
        double b = pos[0] * (is_deg ? gl_deg : 1);
        double l = pos[1] * (is_deg ? gl_deg : 1);
        auto re = Matrix3d(AngleAxisd(-(M_PI / 2 - b), Vector3d::UnitX()) * AngleAxisd(-(M_PI / 2 + l), Vector3d::UnitZ()));
        return re;
    }

    // LLH(默认为经纬度)
    static void SetOrigin(const Eigen::Vector3d pos, bool is_deg = true) {
        _origin = LLH2ECEF(pos, is_deg);
        _cne = Pos2Cne(pos, is_deg);
        _origin_setted = true;
    }

    static Eigen::Vector3d GetOrigin(bool to_deg = true) {
        if (_origin_setted) {
            return ECEF2LLH(_origin, to_deg);
        } else {
            std::cerr << "please set the origin first.\n";
            return Eigen::Vector3d::Zero();
        }
    }

    static Eigen::Vector3d LLH2ENU(const Eigen::Vector3d &pos, const Eigen::Vector3d &origin, bool is_deg = false) {
        Eigen::Vector3d origin_ecef = LLH2ECEF(origin, is_deg);
        Eigen::Matrix3d c_ne = Pos2Cne(origin, is_deg);
        Eigen::Vector3d local_enu = c_ne * (LLH2ECEF(pos, is_deg) - origin_ecef);
        return local_enu;
    }

    static Eigen::Vector3d LLH2ENU(const Eigen::Vector3d &pos, bool is_deg = false) {
        if (_origin_setted) {
            return _cne * (LLH2ECEF(pos, is_deg) - _origin);
        } else {
            std::cerr << "please set the origin first.\n";
            return Eigen::Vector3d::Zero();
        }
    }

    static Eigen::Vector3d ENU2LLH(const Eigen::Vector3d &pos, const Eigen::Vector3d &origin, bool is_deg = false) {
        Eigen::Vector3d origin_ecef = LLH2ECEF(origin, is_deg);
        Eigen::Matrix3d c_ne = Pos2Cne(origin, is_deg);
        Eigen::Vector3d pos_llh = ECEF2LLH(c_ne.transpose() * pos + origin_ecef);
        return pos_llh;
    }

    static Eigen::Vector3d ENU2LLH(const Eigen::Vector3d &pos, bool to_deg = true) {
        if (_origin_setted) {
            return ECEF2LLH(_origin + _cne.transpose() * pos, to_deg);
        } else {
            std::cerr << "please set the origin first.\n";
            return Eigen::Vector3d::Zero();
        }
    }

    static Eigen::Isometry3d Tn0n1(const Eigen::Vector3d &pos, const Eigen::Vector3d &origin) {
        using namespace Eigen;
        Matrix3d const Cn0_e = Pos2Cne(origin);
        Matrix3d const Cn1_e = Pos2Cne(pos);
        Isometry3d const T_n0_n1 = Translation3d(LLH2ENU(pos, origin, false)) * Quaterniond(Cn0_e * Cn1_e.transpose());
        return T_n0_n1;
    }

    static Eigen::Isometry3d Tn0n1(const Eigen::Vector3d &pos) {
        using namespace Eigen;
        Matrix3d const Cn0_e = Pos2Cne(GetOrigin(false));
        Matrix3d const Cn1_e = Pos2Cne(pos);
        Isometry3d const T_n0_n1 = Translation3d(LLH2ENU(pos)) * Quaterniond(Cn0_e * Cn1_e.transpose());
        return T_n0_n1;
    }

    /**
     * @brief 给定中心，范围，返回相应矩形的左上角与右下角点
     *
     * @param pos 中心点
     * @param dis 范围，[-dis, +dis]
     * @return std::pair<Vector3d, Vector3d> 左上角点，右下角点
     */
    std::pair<Eigen::Vector3d, Eigen::Vector3d> static LLHRangeInDistance(Eigen::Vector3d const &pos, const double dis) {
        assert(dis > 0);
        double dlat = dis / M(pos[0]);
        double dlon = dis / (N(pos[0]) * cos(pos[0]));
        Eigen::Vector3d p0 = {pos[0] - dlat, pos[1] - dlon, pos[2]};
        Eigen::Vector3d p1 = {pos[0] + dlat, pos[1] + dlon, pos[2]};
        return {p0, p1};
    }

    /**
     * @brief Get the Gn object 获得当前位置的东北天坐标系下重力矢量
     *
     * @param pos
     * @return Eigen::Vector3d
     */
    static Eigen::Vector3d GetGn(Eigen::Vector3d const &pos) {
        double lat = pos(0);
        double alt = pos(2);
        double sl = sin(lat);
        double gn_u = -(gl_g0 * (1 + 5.27094e-3 * pow(sl, 2) + 2.32718e-5 * pow(sl, 4)) - 3.086e-6 * alt);
        return {0, 0, gn_u};
    }

    static Eigen::Vector3d GetWnie(Eigen::Vector3d const &pos) {
        double lat = pos(0);
        double sl = sin(lat);
        double cl = cos(lat);
        return {0, gl_wie * cl, gl_wie * sl};
    }

    static Eigen::Vector3d GetWnie_back(Eigen::Vector3d const &pos) {
        double lat = pos(0);
        double sl = sin(lat);
        double cl = cos(lat);
        return {0, -gl_wie * cl, -gl_wie * sl};
    }

    static Eigen::Vector3d GetWnen(Eigen::Vector3d const &pos, Eigen::Vector3d const &vel) {
        double lat = pos(0);
        double alt = pos(2);
        double tl = tan(lat);
        double ve = vel[0];
        double vn = vel[1];
        double rmh = M(lat) + alt;
        double rnh = N(lat) + alt;
        return {-vn / rmh, ve / rnh, ve * tl / rnh};
    }
    /**
     * @brief Get the Rm Rn object 获得子午曲率半径，卯酉曲率半径
     *
     * @param pos
     * @return std::pair<double, double>
     */
    static std::pair<double, double> GetRmRn(Eigen::Vector3d const &pos) {
        double b = pos[0];
        return {M(b), N(b)};
    }

    /**
     * @brief 返回pos1-pos2在pos1东北天坐标系下位置
     *
     * @param p1 第一个点大地坐标
     * @param p2 第二个点大地坐标
     * @return Eigen::Vector3d
     */
    static Eigen::Vector3d DeltaPosEnuInFirstPoint(Eigen::Vector3d const &p1, Eigen::Vector3d const &p2) {
        return Pos2Cne(p1) * (LLH2ECEF(p1) - LLH2ECEF(p2));
    }

    /**
     * @brief 返回pos1-pos2在pos2东北天坐标系下位置
     *
     * @param p1 第一个点大地坐标
     * @param p2 第二个点大地坐标
     * @return Eigen::Vector3d
     */
    static Eigen::Vector3d DeltaPosEnuInSecondPoint(Eigen::Vector3d const &p1, Eigen::Vector3d const &p2) {
        return Pos2Cne(p2) * (LLH2ECEF(p1) - LLH2ECEF(p2));
    }

    /**
     * @brief 在pos处添加denu的位置变化
     *
     * @param pos :大地坐标
     * @param denu ：位置变化
     * @return Eigen::Vector3d
     */
    static Eigen::Vector3d PlusDeltaEnuAtPos(Eigen::Vector3d const &pos, Eigen::Vector3d const &denu) {
        return ECEF2LLH(LLH2ECEF(pos) + Pos2Cne(pos).transpose() * denu);
    }

private:
    // double _a, _f, _b, _c, _e1, _e2;         // 长轴 短轴 极点子午曲率半径 扁率 第一偏心率 第二偏心率
    static constexpr double _a = WGS84_RE;
    static constexpr double _f = WGS84_F;
    static constexpr double _b = (1 - _f) * _a;
    static constexpr double _c = _a * _a / _b;
    static constexpr double _e1 = sqrt(_a * _a - _b * _b) / _a;
    static constexpr double _e2 = sqrt(_a * _a - _b * _b) / _b;

    static double W(const double B_) { return sqrt(1 - pow(_e1 * sin(B_), 2)); }
    static double V(const double B_) { return sqrt(1 + pow(_e2 * cos(B_), 2)); }
    static double M(const double B_) { return _c / pow(V(B_), 3); } // 子午曲率半径
    static double N(const double B_) { return _c / V(B_); }         // 卯酉曲率半径

    static Eigen::Vector3d _origin; // ECEF
    static Eigen::Matrix3d _cne;    //
    static bool _origin_setted;     // 是否设置过圆心
};

} // namespace zlam

