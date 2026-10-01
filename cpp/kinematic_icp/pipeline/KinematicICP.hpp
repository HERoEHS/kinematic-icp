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
#include <cmath>
#include <kiss_icp/core/Preprocessing.hpp>
#include <kiss_icp/core/VoxelHashMap.hpp>
#include <sophus/se3.hpp>
#include <tuple>
#include <vector>

#include "kinematic_icp/correspondence_threshold/CorrespondenceThreshold.hpp"
#include "kinematic_icp/registration/Registration.hpp"

namespace kinematic_icp::pipeline {

struct Config {
    // Preprocessing
    double max_range = 100.0;
    double min_range = 0.0;
    // Mapping parameters
    double voxel_size = 1.0;
    unsigned int max_points_per_voxel = 20;
    // Derived parameter, will be computed from other parts of the configuration
    constexpr double map_resolution() const { return voxel_size / std::sqrt(max_points_per_voxel); }
    // Correspondence threshold parameters
    bool use_adaptive_threshold = true;
    double fixed_threshold = 1.0;  // <-- Ignored if use_adaptive_threshold = true

    // Registration Parameters
    int max_num_iterations = 10;
    double convergence_criterion = 0.001;
    int max_num_threads = 1;
    bool use_adaptive_odometry_regularization = true;
    double fixed_regularization = 0.0;  // <-- Ignored if use_adaptive_threshold = true
    // ALICE M2 holonomic mode: estimate (x, y, yaw) instead of (travel, yaw), and
    // optionally weigh the odometry as a Gaussian prior instead of upstream's
    // step damping; see registration/Registration.hpp (HolonomicOptions).
    bool holonomic = false;
    bool odometry_prior = false;
    double regularization_weight_x = 1.0;  // damping mode
    double regularization_weight_y = 1.0;
    double regularization_weight_yaw = 0.0;
    double point_sigma = 0.02;  // prior mode
    double prior_sigma_xy_floor = 0.001;
    double prior_sigma_xy_rel = 0.05;
    double prior_sigma_yaw_floor = 0.0005;
    double prior_sigma_yaw_rel = 0.02;
    bool point_to_line = false;  // holonomic mode only
    double normal_radius = 0.2;
    int normal_min_points = 4;
    double normal_max_eigen_ratio = 0.05;
    double point_to_point_weight = 0.0;
    kinematic_icp::HolonomicOptions holonomic_options() const {
        kinematic_icp::HolonomicOptions h;
        h.enabled = holonomic;
        h.odometry_prior = odometry_prior;
        h.damping_weights = Eigen::Vector3d(regularization_weight_x, regularization_weight_y,
                                            regularization_weight_yaw);
        h.point_sigma = point_sigma;
        h.prior_sigma_xy_floor = prior_sigma_xy_floor;
        h.prior_sigma_xy_rel = prior_sigma_xy_rel;
        h.prior_sigma_yaw_floor = prior_sigma_yaw_floor;
        h.prior_sigma_yaw_rel = prior_sigma_yaw_rel;
        h.point_to_line = point_to_line;
        h.normal_radius = normal_radius;
        h.normal_min_points = normal_min_points;
        h.normal_max_eigen_ratio = normal_max_eigen_ratio;
        h.point_to_point_weight = point_to_point_weight;
        return h;
    }

    // Motion compensation
    bool deskew = false;
};

class KinematicICP {
public:
    using Vector3dVector = std::vector<Eigen::Vector3d>;
    using Vector3dVectorTuple = std::tuple<Vector3dVector, Vector3dVector>;

    explicit KinematicICP(const Config &config)
        : registration_(config.max_num_iterations,
                        config.convergence_criterion,
                        config.max_num_threads,
                        config.use_adaptive_odometry_regularization,
                        config.fixed_regularization,
                        config.holonomic_options()),
          correspondence_threshold_(config.map_resolution(),
                                    config.max_range,
                                    config.use_adaptive_threshold,
                                    config.fixed_threshold),
          config_(config),
          preprocessor_(config.max_range, config.min_range, config.deskew, config.max_num_threads),
          local_map_(config.voxel_size, config.max_range, config.max_points_per_voxel) {}

    Vector3dVectorTuple RegisterFrame(const std::vector<Eigen::Vector3d> &frame,
                                      const std::vector<double> &timestamps,
                                      const Sophus::SE3d &lidar_to_base,
                                      const Sophus::SE3d &relative_odometry) {
        return RegisterFrame(frame, timestamps, lidar_to_base, relative_odometry, relative_odometry);
    }

    // relative_odometry is the robot motion since the previously registered frame
    // (the initial guess); deskew_motion is the robot motion over this scan's own
    // time span (first to last point, timestamps normalized to it). They coincide
    // for one continuously spinning lidar; with two lidars interleaved they do not.
    Vector3dVectorTuple RegisterFrame(const std::vector<Eigen::Vector3d> &frame,
                                      const std::vector<double> &timestamps,
                                      const Sophus::SE3d &lidar_to_base,
                                      const Sophus::SE3d &relative_odometry,
                                      const Sophus::SE3d &deskew_motion);

    const RegistrationDiagnostics &LastRegistrationDiagnostics() const {
        return registration_.diagnostics();
    }

    inline void SetPose(const Sophus::SE3d &pose) {
        last_pose_ = pose;
        local_map_.Clear();
        correspondence_threshold_.Reset();
    };

    std::vector<Eigen::Vector3d> LocalMap() const { return local_map_.Pointcloud(); };

    const kiss_icp::VoxelHashMap &VoxelMap() const { return local_map_; };
    kiss_icp::VoxelHashMap &VoxelMap() { return local_map_; };

    const Sophus::SE3d &pose() const { return last_pose_; }
    Sophus::SE3d &pose() { return last_pose_; }

protected:
    Sophus::SE3d last_pose_;
    // Kinematic module
    KinematicRegistration registration_;
    CorrespondenceThreshold correspondence_threshold_;
    Config config_;
    // KISS-ICP pipeline modules
    kiss_icp::Preprocessor preprocessor_;
    kiss_icp::VoxelHashMap local_map_;
};

}  // namespace kinematic_icp::pipeline
