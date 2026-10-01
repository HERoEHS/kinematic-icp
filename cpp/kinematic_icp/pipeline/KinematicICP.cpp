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
#include "KinematicICP.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <kiss_icp/core/Preprocessing.hpp>
#include <kiss_icp/core/VoxelHashMap.hpp>
#include <limits>
#include <vector>

namespace {
auto transform_points(const std::vector<Eigen::Vector3d> &points, const Sophus::SE3d &pose) {
    std::vector<Eigen::Vector3d> points_transformed(points.size());
    std::transform(points.cbegin(), points.cend(), points_transformed.begin(),
                   [&](const auto &point) { return pose * point; });
    return points_transformed;
}

auto Voxelize(const std::vector<Eigen::Vector3d> &frame, const double voxel_size) {
    const std::vector<Eigen::Vector3d> &frame_downsample =
        kiss_icp::VoxelDownsample(frame, voxel_size * 0.5);
    const std::vector<Eigen::Vector3d> &source =
        kiss_icp::VoxelDownsample(frame_downsample, voxel_size * 1.5);
    return std::make_tuple(source, frame_downsample);
}

double NearestDistance2D(const std::vector<Eigen::Vector3d> &cloud, const Eigen::Vector3d &p) {
    double best = std::numeric_limits<double>::max();
    for (const auto &q : cloud) best = std::min(best, (q.head<2>() - p.head<2>()).squaredNorm());
    return std::sqrt(best);
}
}  // namespace
namespace kinematic_icp::pipeline {

std::vector<bool> KinematicICP::MovingWithRobot(const std::vector<Eigen::Vector3d> &frame,
                                                const int sensor,
                                                const Sophus::SE3d &pose) const {
    std::vector<bool> moving(frame.size(), false);
    // The newest earlier frame of this lidar that the robot has travelled far enough
    // from to tell a point that followed it from one that stayed. Travel only: turning
    // on the spot does not drag the odometry along, and judging then costs points.
    const RecentFrame *reference = nullptr;
    for (auto it = recent_frames_.crbegin(); it != recent_frames_.crend(); ++it) {
        if (it->sensor != sensor) continue;
        const Sophus::SE3d motion = it->pose.inverse() * pose;
        if (motion.translation().head<2>().norm() >= config_.moving_object_min_motion) {
            reference = &(*it);
            break;
        }
    }
    if (reference == nullptr) return moving;
    const Sophus::SE3d current_to_reference = reference->pose.inverse() * pose;
    const Sophus::SE3d base_to_lidar = reference->lidar_to_base.inverse();
    // The robot body: the configured box, else the box spanned by the lidar mounts.
    std::vector<double> body = config_.moving_object_footprint;
    if (body.size() != 4 && lidar_positions_.size() >= 2) {
        Eigen::Vector2d lo = lidar_positions_.cbegin()->second, hi = lo;
        for (const auto &[s, position] : lidar_positions_) {
            lo = lo.cwiseMin(position);
            hi = hi.cwiseMax(position);
        }
        if ((hi - lo).minCoeff() > 0.1) body = {lo.x(), hi.x(), lo.y(), hi.y()};
    }
    auto inside_robot = [&](const Eigen::Vector3d &q) {
        return body.size() == 4 && q.x() > body[0] && q.x() < body[1] && q.y() > body[2] &&
               q.y() < body[3];
    };
    const auto blind_it = blind_sectors_.find(sensor);
    auto unseen = [&](const Eigen::Vector3d &q) {
        if (blind_it == blind_sectors_.cend()) return false;
        const Eigen::Vector3d l = base_to_lidar * q;
        const double a = std::atan2(l.y(), l.x()) * 180.0 / M_PI;
        const auto &sectors = blind_it->second;
        for (std::size_t k = 0; k + 1 < sectors.size(); k += 2) {
            const double from = sectors[k], to = sectors[k + 1];
            if (from <= to ? (a >= from && a <= to) : (a >= from || a <= to)) return true;
        }
        return false;
    };
    for (std::size_t i = 0; i < frame.size(); ++i) {
        const Eigen::Vector3d &p = frame[i];
        if (NearestDistance2D(reference->points, p) > config_.moving_object_same_tolerance) continue;
        // Where the point was then, had it stayed put.
        const Eigen::Vector3d q = current_to_reference * p;
        if (inside_robot(q)) {
            moving[i] = true;
        } else if (!unseen(q)) {
            moving[i] = NearestDistance2D(reference->points, q) > config_.moving_object_static_tolerance;
        }
    }
    return moving;
}

KinematicICP::Vector3dVectorTuple KinematicICP::RegisterFrame(
    const std::vector<Eigen::Vector3d> &frame,
    const std::vector<double> &timestamps,
    const Sophus::SE3d &lidar_to_base,
    const Sophus::SE3d &relative_odometry,
    const Sophus::SE3d &deskew_motion,
    const int sensor) {
    // Need to deskew in lidar frame
    const Sophus::SE3d &deskew_motion_in_lidar =
        lidar_to_base.inverse() * deskew_motion * lidar_to_base;
    const auto &preprocessed_frame =
        preprocessor_.Preprocess(frame, timestamps, deskew_motion_in_lidar);
    // Give the frame in base frame
    const auto &preprocessed_frame_in_base = transform_points(preprocessed_frame, lidar_to_base);

    // ALICE M2: leave out what moved with the robot, see Config::moving_object_filter.
    std::vector<Eigen::Vector3d> fixed_points;
    const std::vector<Eigen::Vector3d> *registered_points = &preprocessed_frame_in_base;
    last_num_moving_points_ = 0;
    lidar_positions_[sensor] = lidar_to_base.translation().head<2>();
    if (config_.moving_object_filter) {
        const auto moving =
            MovingWithRobot(preprocessed_frame_in_base, sensor, last_pose_ * relative_odometry);
        fixed_points.reserve(preprocessed_frame_in_base.size());
        for (std::size_t i = 0; i < moving.size(); ++i) {
            if (!moving[i]) fixed_points.push_back(preprocessed_frame_in_base[i]);
        }
        last_num_moving_points_ = preprocessed_frame_in_base.size() - fixed_points.size();
        registered_points = &fixed_points;
    }

    // Voxelize
    const auto &[source, frame_downsample] = Voxelize(*registered_points, config_.voxel_size);

    // Get adaptive_threshold
    const double tau = correspondence_threshold_.ComputeThreshold();

    // Run ICP
    // Holonomic point-to-line: normals of the registration points from the full
    // deskewed frame (robot frame).
    const std::vector<Eigen::Vector3d> normals =
        (config_.holonomic && config_.point_to_line)
            ? registration_.EstimateNormals(source, *registered_points)
            : std::vector<Eigen::Vector3d>{};
    const auto new_pose = registration_.ComputeRobotMotion(source,             // frame
                                                           normals,            // normals
                                                           local_map_,         // voxel_map
                                                           last_pose_,         // last_pose
                                                           relative_odometry,  // robot_motion
                                                           tau);  // max_correspondence_dist

    // Compute the difference between the prediction and the actual estimate
    const auto odometry_error = (last_pose_ * relative_odometry).inverse() * new_pose;

    // Update step: threshold, local map and the last pose
    correspondence_threshold_.UpdateOdometryError(odometry_error);
    local_map_.Update(frame_downsample, new_pose);
    last_pose_ = new_pose;
    if (config_.moving_object_filter) {
        // Unfiltered, so whatever follows the robot is still there to be recognised.
        recent_frames_.push_back({sensor, new_pose, lidar_to_base, preprocessed_frame_in_base});
        while (recent_frames_.size() > static_cast<std::size_t>(std::max(1, config_.moving_object_history))) {
            recent_frames_.pop_front();
        }
    }

    // Return the (deskew) input raw scan (frame) and the points used for
    // registration (source)
    return {preprocessed_frame_in_base, source};
}
}  // namespace kinematic_icp::pipeline
