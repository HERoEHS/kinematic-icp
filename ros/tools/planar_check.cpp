// ALICE M2: synthetic check of the Kinematic-ICP pipeline on holonomic motion.
//
// No ROS. A 2D world of line segments is ray-cast by a simulated 270 deg / 10 Hz
// lidar with per-beam timing, while the robot follows a holonomic trajectory
// (forward, crab, spin, diagonal, arc). The odometry prior handed to the
// pipeline mimics the M2 wheel+gyro EKF: heading nearly exact (gyro), travel
// distorted by wheel errors (+0.8 % along x, -4 % along y while crabbing, i.e.
// lateral slip). The estimate is compared with ground truth and with the prior
// alone.
//
//   kinematic_icp_planar_check [key=value ...]
//   keys: voxel holonomic prior w_xy w_yaw point_sigma xy_rel xy_floor yaw_rel yaw_floor
//         p2l normal_radius normal_ratio p2p_weight seed
#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <sophus/se3.hpp>

#include "kinematic_icp/pipeline/KinematicICP.hpp"

namespace {

struct Seg {
    Eigen::Vector2d a, b;
};
struct Pose2 {
    double x{0.0}, y{0.0}, th{0.0};
};
struct Phase {
    const char *name;
    double duration, vx, vy, wz;
};

double Cross(const Eigen::Vector2d &a, const Eigen::Vector2d &b) { return a.x() * b.y() - a.y() * b.x(); }
double Wrap(double a) { return std::atan2(std::sin(a), std::cos(a)); }

Sophus::SE3d ToSE3(const Pose2 &p) {
    return Sophus::SE3d(Sophus::SO3d::rotZ(p.th), Eigen::Vector3d(p.x, p.y, 0.0));
}
Pose2 FromSE3(const Sophus::SE3d &T) {
    const Eigen::Matrix3d R = T.rotationMatrix();
    return {T.translation().x(), T.translation().y(), std::atan2(R(1, 0), R(0, 0))};
}

void Box(std::vector<Seg> &w, double x0, double y0, double x1, double y1) {
    w.push_back({{x0, y0}, {x1, y0}});
    w.push_back({{x1, y0}, {x1, y1}});
    w.push_back({{x1, y1}, {x0, y1}});
    w.push_back({{x0, y1}, {x0, y0}});
}

std::vector<Seg> MakeHall() {
    std::vector<Seg> w;
    Box(w, 0.0, 0.0, 16.0, 12.0);
    Box(w, 8.8, 3.8, 9.2, 4.2);
    Box(w, 10.8, 8.8, 11.2, 9.2);
    Box(w, 12.8, 4.8, 13.2, 5.2);
    Box(w, 2.8, 9.8, 3.2, 10.2);
    Box(w, 14.0, 1.0, 15.0, 2.0);
    w.push_back({{7.0, 11.0}, {11.0, 11.0}});
    return w;
}

// 1.0 m corridor, 42 m long, with shallow door recesses every 7 m on
// alternating sides - the case where only lateral offset and heading are
// well observed.
std::vector<Seg> MakeCorridor() {
    std::vector<Seg> w;
    const double width = 1.0, depth = 0.15, door = 0.9;
    for (int side = 0; side < 2; ++side) {
        const double y = side == 0 ? 0.0 : width;
        const double out = side == 0 ? -depth : depth;
        double x = -2.0;
        for (double d = 5.0 + 3.5 * side; d < 40.0; d += 7.0) {
            w.push_back({{x, y}, {d, y}});
            w.push_back({{d, y}, {d, y + out}});
            w.push_back({{d, y + out}, {d + door, y + out}});
            w.push_back({{d + door, y + out}, {d + door, y}});
            x = d + door;
        }
        w.push_back({{x, y}, {40.0, y}});
    }
    w.push_back({{-2.0, 0.0}, {-2.0, width}});
    w.push_back({{40.0, 0.0}, {40.0, width}});
    return w;
}

double Raycast(const std::vector<Seg> &w, const Eigen::Vector2d &o, const Eigen::Vector2d &dir, double rmax) {
    double best = std::numeric_limits<double>::infinity();
    for (const auto &s : w) {
        const Eigen::Vector2d e = s.b - s.a;
        const double den = Cross(dir, e);
        if (std::abs(den) < 1e-12) continue;
        const Eigen::Vector2d d = s.a - o;
        const double t = Cross(d, e) / den;
        const double u = Cross(d, dir) / den;
        if (t > 1e-6 && u >= 0.0 && u <= 1.0 && t < best) best = t;
    }
    return best <= rmax ? best : -1.0;
}

// Ground truth at 1 ms from a phase list, with 0.3 s linear ramps between phases.
std::vector<Pose2> Integrate(const std::vector<Phase> &phases, const Pose2 &start, double dt,
                             std::vector<int> &phase_of_step) {
    std::vector<Pose2> P{start};
    phase_of_step.push_back(0);
    double vx = 0, vy = 0, wz = 0;
    const double ramp = 0.3;
    for (size_t k = 0; k < phases.size(); ++k) {
        const auto &ph = phases[k];
        const int n = static_cast<int>(std::lround(ph.duration / dt));
        const double vx0 = vx, vy0 = vy, wz0 = wz;
        for (int i = 0; i < n; ++i) {
            const double a = std::min(1.0, (i * dt) / ramp);
            vx = vx0 + a * (ph.vx - vx0);
            vy = vy0 + a * (ph.vy - vy0);
            wz = wz0 + a * (ph.wz - wz0);
            Pose2 p = P.back();
            p.x += (vx * std::cos(p.th) - vy * std::sin(p.th)) * dt;
            p.y += (vx * std::sin(p.th) + vy * std::cos(p.th)) * dt;
            p.th = Wrap(p.th + wz * dt);
            P.push_back(p);
            phase_of_step.push_back(static_cast<int>(k));
        }
    }
    return P;
}

struct Result {
    double final_pos, final_yaw, rms_pos, max_pos;
    std::vector<double> phase_end_pos, phase_end_yaw;
};

Result Score(const std::vector<Pose2> &est, const std::vector<Pose2> &gt, const std::vector<int> &frame_phase,
             int n_phases) {
    Result r{0, 0, 0, 0, std::vector<double>(n_phases, 0.0), std::vector<double>(n_phases, 0.0)};
    double s = 0;
    for (size_t k = 0; k < est.size(); ++k) {
        const double e = std::hypot(est[k].x - gt[k].x, est[k].y - gt[k].y);
        s += e * e;
        r.max_pos = std::max(r.max_pos, e);
        r.phase_end_pos[frame_phase[k]] = e;
        r.phase_end_yaw[frame_phase[k]] = Wrap(est[k].th - gt[k].th) * 180.0 / M_PI;
    }
    r.rms_pos = std::sqrt(s / est.size());
    r.final_pos = std::hypot(est.back().x - gt.back().x, est.back().y - gt.back().y);
    r.final_yaw = Wrap(est.back().th - gt.back().th) * 180.0 / M_PI;
    return r;
}

void Run(const char *name, const std::vector<Seg> &world, const std::vector<Phase> &phases, const Pose2 &start,
         const kinematic_icp::pipeline::Config &base_cfg, unsigned seed) {
    const double dt = 0.001, scan_period = 0.1, rmax = 12.0, sigma_r = 0.015;
    const int n_beams = 301;  // 270 deg at 0.9 deg
    const double fov_min = -135.0 * M_PI / 180.0, fov_max = 135.0 * M_PI / 180.0;
    const Pose2 mount{0.39, 0.345, M_PI / 4.0};  // front-left corner, as in the sim xacro
    const Sophus::SE3d lidar_to_base = ToSE3(mount);

    std::vector<int> phase_of_step;
    const std::vector<Pose2> P = Integrate(phases, start, dt, phase_of_step);
    const int steps_per_scan = static_cast<int>(std::lround(scan_period / dt));

    kinematic_icp::pipeline::Config cfg = base_cfg;
    cfg.max_range = rmax;
    cfg.min_range = 0.05;
    cfg.deskew = true;
    cfg.max_num_threads = 1;
    kinematic_icp::pipeline::KinematicICP kicp(cfg);
    kicp.SetPose(ToSE3(P.front()));

    std::mt19937 rng(seed);
    std::normal_distribution<double> nr(0.0, sigma_r), nt(0.0, 0.002), nth(0.0, 0.0002);

    std::vector<Pose2> gt, est, prior;
    std::vector<int> frame_phase;
    Pose2 prior_pose = P.front();
    double t_reg = 0.0;
    int frames = 0;
    for (size_t k = steps_per_scan; k < P.size(); k += steps_per_scan) {
        // Scan covering (k - steps_per_scan, k], beam i at its own robot pose.
        std::vector<Eigen::Vector3d> pts;
        std::vector<double> stamps;
        for (int i = 0; i < n_beams; ++i) {
            const double f = static_cast<double>(i) / (n_beams - 1);
            const size_t step = k - steps_per_scan + static_cast<size_t>(std::lround(f * steps_per_scan));
            const Pose2 &rp = P[step];
            const Pose2 sp = FromSE3(ToSE3(rp) * lidar_to_base);
            const double a = fov_min + f * (fov_max - fov_min);
            const Eigen::Vector2d dir(std::cos(sp.th + a), std::sin(sp.th + a));
            const double range = Raycast(world, {sp.x, sp.y}, dir, rmax);
            if (range < 0.0) continue;
            const double rn = range + nr(rng);
            pts.emplace_back(rn * std::cos(a), rn * std::sin(a), 0.0);
            stamps.push_back(f);
        }
        // Prior: true relative motion with wheel-like errors.
        const Pose2 d = FromSE3(ToSE3(P[k - steps_per_scan]).inverse() * ToSE3(P[k]));
        const double vy_cmd = phases[phase_of_step[k]].vy;
        const double ex = 0.008, ey = std::abs(vy_cmd) > 0.05 ? -0.04 : 0.008, eth = 0.003;
        const Pose2 dn{d.x * (1.0 + ex) + nt(rng), d.y * (1.0 + ey) + nt(rng), d.th * (1.0 + eth) + nth(rng)};
        const Sophus::SE3d delta = ToSE3(dn);
        prior_pose = FromSE3(ToSE3(prior_pose) * delta);

        const auto t0 = std::chrono::steady_clock::now();
        kicp.RegisterFrame(pts, stamps, lidar_to_base, delta);
        t_reg += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        ++frames;

        gt.push_back(P[k]);
        est.push_back(FromSE3(kicp.pose()));
        prior.push_back(prior_pose);
        frame_phase.push_back(phase_of_step[k]);
    }
    const int np = static_cast<int>(phases.size());
    const Result rp = Score(prior, gt, frame_phase, np);
    const Result re = Score(est, gt, frame_phase, np);
    std::printf("\n== %s  (voxel %.2f, %s, %d frames, %.2f ms/frame)\n", name, cfg.voxel_size,
                !cfg.holonomic ? "unicycle (original)"
                               : (cfg.odometry_prior ? (cfg.point_to_line ? "HOLONOMIC prior p2l" : "HOLONOMIC prior")
                                                     : "HOLONOMIC damping"),
                frames, 1e3 * t_reg / frames);
    std::printf("  %-14s | %-22s | %-22s\n", "phase end", "prior: pos[cm] yaw[deg]", "kicp: pos[cm] yaw[deg]");
    for (int p = 0; p < np; ++p) {
        std::printf("  %-14s | %10.1f %10.2f | %10.1f %10.2f\n", phases[p].name, 100 * rp.phase_end_pos[p],
                    rp.phase_end_yaw[p], 100 * re.phase_end_pos[p], re.phase_end_yaw[p]);
    }
    std::printf("  %-14s | %10.1f %10.2f | %10.1f %10.2f\n", "FINAL", 100 * rp.final_pos, rp.final_yaw,
                100 * re.final_pos, re.final_yaw);
    std::printf("  %-14s | %10.1f %10s | %10.1f %10s\n", "rms / max", 100 * rp.rms_pos, "", 100 * re.rms_pos, "");
    std::printf("  %-14s | %10.1f %10s | %10.1f %10s\n", "", 100 * rp.max_pos, "", 100 * re.max_pos, "");
}

}  // namespace

int main(int argc, char **argv) {
    kinematic_icp::pipeline::Config cfg;
    unsigned seed = 42;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto eq = a.find('=');
        if (eq == std::string::npos) continue;
        const std::string k = a.substr(0, eq);
        const double v = std::atof(a.c_str() + eq + 1);
        if (k == "seed") seed = static_cast<unsigned>(v);
        else if (k == "voxel") cfg.voxel_size = v;
        else if (k == "holonomic") cfg.holonomic = v != 0.0;
        else if (k == "prior") cfg.odometry_prior = v != 0.0;
        else if (k == "w_xy") cfg.regularization_weight_x = cfg.regularization_weight_y = v;
        else if (k == "w_yaw") cfg.regularization_weight_yaw = v;
        else if (k == "point_sigma") cfg.point_sigma = v;
        else if (k == "xy_rel") cfg.prior_sigma_xy_rel = v;
        else if (k == "xy_floor") cfg.prior_sigma_xy_floor = v;
        else if (k == "yaw_rel") cfg.prior_sigma_yaw_rel = v;
        else if (k == "yaw_floor") cfg.prior_sigma_yaw_floor = v;
        else if (k == "p2l") cfg.point_to_line = v != 0.0;
        else if (k == "normal_radius") cfg.normal_radius = v;
        else if (k == "normal_ratio") cfg.normal_max_eigen_ratio = v;
        else if (k == "p2p_weight") cfg.point_to_point_weight = v;
        else std::printf("unknown key %s\n", k.c_str());
    }
    const std::vector<Phase> hall = {
        {"forward", 5.0, 0.5, 0.0, 0.0},        {"crab +y", 5.0, 0.0, 0.5, 0.0},
        {"spin +90", M_PI, 0.0, 0.0, 0.5},       {"diagonal", 6.0, 0.35, 0.35, 0.0},
        {"crab -y", 6.0, 0.0, -0.5, 0.0},       {"arc", 6.0, 0.4, 0.0, -0.3},
    };
    Run("hall: mixed holonomic motion", MakeHall(), hall, {2.5, 2.5, 0.0}, cfg, seed);
    const std::vector<Phase> corridor = {
        {"forward 12 m", 24.0, 0.5, 0.0, 0.0},
        {"reverse 5 m", 10.0, -0.5, 0.0, 0.0},
    };
    Run("corridor 1.0 m: forward / reverse", MakeCorridor(), corridor, {1.0, 0.5, 0.0}, cfg, seed);
    return 0;
}
