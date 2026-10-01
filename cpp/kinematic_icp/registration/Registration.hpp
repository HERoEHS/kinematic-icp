// MIT License

// Copyright (c) 2024 Tiziano Guadagnino, Benedikt Mersch, Ignacio Vizzo, Cyrill
// Stachniss.

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include <Eigen/Core>
#include <kiss_icp/core/VoxelHashMap.hpp>
#include <sophus/se3.hpp>
#include <vector>

namespace kinematic_icp {

// Holonomic mode (ALICE M2, 2026-10-01). Upstream solves for two parameters,
// travel along the robot x axis and yaw (unicycle), so a lateral component of the
// prior can never be corrected: in a crab the lidar inherits the wheels' lateral
// slip, and the misfit leaks into yaw. With enabled set, the solver estimates the
// full planar increment (x, y, yaw). Two ways to weigh the odometry:
//
//  - damping (odometry_prior false): upstream's scheme extended per axis,
//    beta * damping_weights added to J^T J at every Gauss-Newton step. It shrinks
//    each step, it does not pull towards the prior. Synthetic check
//    (tools/planar_check.cpp): weight 1 cannot correct a 4 % crab slip (it sits
//    below the residual floor, so the steps are damped to nothing and the loop
//    "converges" at the prior); weight 0.1 corrects it but lets the estimate run
//    6 m along a 1.0 m corridor, where nothing anchors the weak direction.
//  - prior (odometry_prior true): MAP with the odometry increment as a Gaussian
//    on (x, y, yaw). Per frame, sigma = floor + rel * |motion on that axis|; the
//    lidar term is sum |r|^2 / point_sigma^2. Normalized by N/point_sigma^2 like
//    the J^T J above, the prior adds lambda = point_sigma^2 / (N sigma^2) both to
//    J^T J and, times the current deviation from the prior, to J^T r. A direction
//    the scan constrains follows the scan; a degenerate one stays on the prior.
//
// Point-to-line residuals (point_to_line). Point-to-point ICP has a translation
// block of J^T J that is the identity whatever the geometry, so a corridor looks
// fully constrained along its axis and the prior above cannot tell it is not:
// synthetic 1.0 m corridor, prior mode, the estimate stuck to the map and ended
// 1.5..6 m short. With a normal per source point (fitted to the frame's own
// points within normal_radius, kept only if the neighbourhood is line-like), the
// residual is n . (T s - t) and the along-wall direction carries no information,
// so it stays on the prior. Points without a valid normal (corners, clutter)
// enter as point-to-point with weight point_to_point_weight (0 drops them).
//
// With enabled false the original two-parameter code path runs unchanged.
struct HolonomicOptions {
    bool enabled = false;
    bool odometry_prior = false;
    bool point_to_line = false;
    double normal_radius = 0.2;            // [m]
    int normal_min_points = 4;
    double normal_max_eigen_ratio = 0.05;  // lambda_min / lambda_max of the 2D scatter
    double point_to_point_weight = 0.0;
    Eigen::Vector3d damping_weights{1.0, 1.0, 0.0};  // (x, y, yaw); upstream is (1, -, 0)
    double point_sigma = 0.02;                       // [m] per correspondence
    double prior_sigma_xy_floor = 0.001;             // [m] per frame
    double prior_sigma_xy_rel = 0.05;                // fraction of |dx| resp. |dy|
    double prior_sigma_yaw_floor = 0.0005;           // [rad] per frame
    double prior_sigma_yaw_rel = 0.02;               // fraction of |dyaw|
};

struct RegistrationDiagnostics {
    // Normalized Gauss-Newton information (J^T J / N + Omega) in (x, y, yaw) at the
    // returned estimate, the mean squared point residual there, the number of
    // correspondences, and the diagonal of the odometry term Omega (beta *
    // weights in damping mode, lambda in prior mode).
    Eigen::Matrix3d information = Eigen::Matrix3d::Zero();
    double mean_squared_residual = 0.0;
    std::size_t num_correspondences = 0;
    Eigen::Vector3d omega_diagonal = Eigen::Vector3d::Zero();
    std::size_t num_point_to_line = 0;  // correspondences that used a normal
};

struct KinematicRegistration {
    explicit KinematicRegistration(const int max_num_iteration,
                                   const double convergence_criterion,
                                   const int max_num_threads,
                                   const bool use_adaptive_odometry_regularization,
                                   const double fixed_regularization,
                                   const HolonomicOptions &holonomic = HolonomicOptions{});

    Sophus::SE3d ComputeRobotMotion(const std::vector<Eigen::Vector3d> &frame,
                                    const kiss_icp::VoxelHashMap &voxel_map,
                                    const Sophus::SE3d &last_robot_pose,
                                    const Sophus::SE3d &relative_wheel_odometry,
                                    const double max_correspondence_distance) {
        return ComputeRobotMotion(frame, {}, voxel_map, last_robot_pose, relative_wheel_odometry,
                                  max_correspondence_distance);
    }

    // normals: one per frame point in the robot frame (zero = no valid normal), or
    // empty for point-to-point. Used only in holonomic mode with point_to_line.
    Sophus::SE3d ComputeRobotMotion(const std::vector<Eigen::Vector3d> &frame,
                                    const std::vector<Eigen::Vector3d> &normals,
                                    const kiss_icp::VoxelHashMap &voxel_map,
                                    const Sophus::SE3d &last_robot_pose,
                                    const Sophus::SE3d &relative_wheel_odometry,
                                    const double max_correspondence_distance);

    // Line normals of `points` from the neighbourhood in `cloud` (both in the
    // robot frame), per the point_to_line options. Zero where not line-like.
    std::vector<Eigen::Vector3d> EstimateNormals(const std::vector<Eigen::Vector3d> &points,
                                                 const std::vector<Eigen::Vector3d> &cloud) const;

    const RegistrationDiagnostics &diagnostics() const { return diagnostics_; }

    int max_num_iterations_;
    double convergence_criterion_;
    int max_num_threads_;
    bool use_adaptive_odometry_regularization_;
    double fixed_regularization_;
    HolonomicOptions holonomic_;
    RegistrationDiagnostics diagnostics_;
};
}  // namespace kinematic_icp
