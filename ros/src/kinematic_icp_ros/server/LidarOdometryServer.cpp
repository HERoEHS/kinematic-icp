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
#include "kinematic_icp_ros/server/LidarOdometryServer.hpp"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>
#include <sophus/se3.hpp>
#include <utility>

#include "kinematic_icp/pipeline/KinematicICP.hpp"
#include "kinematic_icp_ros/utils/RosUtils.hpp"

// ROS 2 headers
#include <rcl/time.h>
#include <tf2_ros/buffer_interface.h>
#include <tf2_ros/transform_broadcaster.h>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <visualization_msgs/msg/marker.hpp>

namespace {
// Time Type aliases to simplify code
using milliseconds = std::chrono::milliseconds;
using seconds = std::chrono::duration<long double>;
using std::chrono::duration_cast;
}  // namespace

namespace kinematic_icp_ros {

using namespace utils;

LidarOdometryServer::LidarOdometryServer(rclcpp::Node::SharedPtr node) : node_(node) {
    lidar_odom_frame_ = node->declare_parameter<std::string>("lidar_odom_frame", lidar_odom_frame_);
    wheel_odom_frame_ = node->declare_parameter<std::string>("wheel_odom_frame", wheel_odom_frame_);
    base_frame_ = node->declare_parameter<std::string>("base_frame", base_frame_);
    publish_odom_tf_ = node->declare_parameter<bool>("publish_odom_tf", publish_odom_tf_);
    invert_odom_tf_ = node->declare_parameter<bool>("invert_odom_tf", invert_odom_tf_);
    tf_timeout_ =
        duration_cast<milliseconds>(seconds(node->declare_parameter<double>("tf_timeout", 0.0)));

    kinematic_icp::pipeline::Config config;
    // Preprocessing
    config.max_range = node->declare_parameter<double>("max_range", config.max_range);
    config.min_range = node->declare_parameter<double>("min_range", config.min_range);
    // Mapping parameters
    config.voxel_size = node->declare_parameter<double>("voxel_size", config.voxel_size);
    config.max_points_per_voxel =
        node->declare_parameter<int>("max_points_per_voxel", config.max_points_per_voxel);
    // Correspondence threshold parameters
    config.use_adaptive_threshold =
        node->declare_parameter<bool>("use_adaptive_threshold", config.use_adaptive_threshold);
    config.fixed_threshold =
        node->declare_parameter<double>("fixed_threshold", config.fixed_threshold);
    // Registration Parameters
    config.max_num_iterations =
        node->declare_parameter<int>("max_num_iterations", config.max_num_iterations);
    config.convergence_criterion =
        node->declare_parameter<double>("convergence_criterion", config.convergence_criterion);
    config.max_num_threads =
        node->declare_parameter<int>("max_num_threads", config.max_num_threads);
    config.use_adaptive_odometry_regularization = node->declare_parameter<bool>(
        "use_adaptive_odometry_regularization", config.use_adaptive_odometry_regularization);
    config.fixed_regularization =
        node->declare_parameter<double>("fixed_regularization", config.fixed_regularization);
    // Motion compensation
    config.deskew = node->declare_parameter<bool>("deskew", config.deskew);
    // ALICE M2 holonomic options, see cpp/kinematic_icp/registration/Registration.hpp.
    config.holonomic = node->declare_parameter<bool>("holonomic", config.holonomic);
    config.odometry_prior = node->declare_parameter<bool>("odometry_prior", config.odometry_prior);
    config.regularization_weight_x =
        node->declare_parameter<double>("regularization_weight_x", config.regularization_weight_x);
    config.regularization_weight_y =
        node->declare_parameter<double>("regularization_weight_y", config.regularization_weight_y);
    config.regularization_weight_yaw = node->declare_parameter<double>(
        "regularization_weight_yaw", config.regularization_weight_yaw);
    config.point_sigma = node->declare_parameter<double>("point_sigma", config.point_sigma);
    config.prior_sigma_xy_floor =
        node->declare_parameter<double>("prior_sigma_xy_floor", config.prior_sigma_xy_floor);
    config.prior_sigma_xy_rel =
        node->declare_parameter<double>("prior_sigma_xy_rel", config.prior_sigma_xy_rel);
    config.prior_sigma_yaw_floor =
        node->declare_parameter<double>("prior_sigma_yaw_floor", config.prior_sigma_yaw_floor);
    config.prior_sigma_yaw_rel =
        node->declare_parameter<double>("prior_sigma_yaw_rel", config.prior_sigma_yaw_rel);
    config.point_to_line = node->declare_parameter<bool>("point_to_line", config.point_to_line);
    config.normal_radius = node->declare_parameter<double>("normal_radius", config.normal_radius);
    config.normal_min_points =
        node->declare_parameter<int>("normal_min_points", config.normal_min_points);
    config.normal_max_eigen_ratio =
        node->declare_parameter<double>("normal_max_eigen_ratio", config.normal_max_eigen_ratio);
    config.point_to_point_weight =
        node->declare_parameter<double>("point_to_point_weight", config.point_to_point_weight);
    covariance_from_registration_ =
        node->declare_parameter<bool>("covariance_from_registration", covariance_from_registration_);
    covariance_scale_ = node->declare_parameter<double>("covariance_scale", covariance_scale_);
    if (config.max_range < config.min_range) {
        RCLCPP_WARN(node_->get_logger(),
                    "[WARNING] max_range is smaller than min_range, settng min_range to 0.0");
        config.min_range = 0.0;
    }

    // Construct the main KISS-ICP odometry node
    config_ = config;
    kinematic_icp_ = std::make_unique<kinematic_icp::pipeline::KinematicICP>(config);

    // Initialize publishers
    rclcpp::QoS qos((rclcpp::SystemDefaultsQoS().keep_last(1).durability_volatile()));
    odom_publisher_ = node_->create_publisher<nav_msgs::msg::Odometry>("lidar_odometry", qos);
    frame_publisher_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("frame", qos);
    kpoints_publisher_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("keypoints", qos);
    map_publisher_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("local_map", qos);
    voxel_grid_pub_ = node_->create_publisher<visualization_msgs::msg::Marker>("voxel_grid", qos);
    // ALICE M2: per-frame registration diagnostics, layout in RegisterFrame().
    registration_debug_pub_ =
        node_->create_publisher<std_msgs::msg::Float64MultiArray>("registration_debug", qos);

    set_pose_srv_ = node_->create_service<std_srvs::srv::Trigger>(
        "set_pose", [&](const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                        std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
            const auto pose = LookupTransform(wheel_odom_frame_, base_frame_, tf2_buffer_);
            RCLCPP_WARN_STREAM(node_->get_logger(), "Resetting KISS-ICP pose:\n"
                                                        << pose.matrix() << "\n");
            kinematic_icp_->SetPose(pose);
            response->success = true;
        });

    // Initialize the transform broadcaster
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(node_);
    tf2_buffer_ = std::make_unique<tf2_ros::Buffer>(node_->get_clock());
    tf2_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf2_buffer_);

    // Initialize the odometry and tf msg. Since tf by design does not allow for having two frames
    // in the tree with a different parent, to publish the odom_lidar frame, we need to add a child
    // frame (we can't provide odom_lidar -> base_footprint, which is what we estimate internally).
    // Therefore we publish the inverted transformation (base_footprint -> odom_lidar), which turns
    // to be an implemantation detail as tf lookups are transparent and one can query for the
    // odom_lidar -> base_footprint transform, the tf will invert it again on the fly.
    if (invert_odom_tf_) {
        tf_msg_.header.frame_id = base_frame_;
        tf_msg_.child_frame_id = lidar_odom_frame_;
    } else {
        tf_msg_.header.frame_id = lidar_odom_frame_;
        tf_msg_.child_frame_id = base_frame_;
    }

    // fixed covariancecovariance
    position_covariance_ = node->declare_parameter<double>("position_covariance", 0.1);
    orientation_covariance_ = node->declare_parameter<double>("orientation_covariance", 0.1);
    odom_msg_.header.frame_id = lidar_odom_frame_;
    odom_msg_.child_frame_id = base_frame_;
    odom_msg_.pose.covariance.fill(0.0);
    odom_msg_.pose.covariance[0] = position_covariance_;
    odom_msg_.pose.covariance[7] = position_covariance_;
    odom_msg_.pose.covariance[35] = orientation_covariance_;
    odom_msg_.twist.covariance.fill(0);
    odom_msg_.twist.covariance[0] = position_covariance_;
    odom_msg_.twist.covariance[7] = position_covariance_;
    odom_msg_.twist.covariance[35] = orientation_covariance_;
}

bool LidarOdometryServer::LookupExtrinsic(const std::string &sensor_frame, Sophus::SE3d &extrinsic) {
    const auto it = sensor_to_base_footprint_.find(sensor_frame);
    if (it != sensor_to_base_footprint_.end()) {
        extrinsic = it->second;
        return true;
    }
    try {
        extrinsic = tf2::transformToSophus(
            tf2_buffer_->lookupTransform(base_frame_, sensor_frame, tf2::TimePointZero));
    } catch (tf2::TransformException &ex) {
        RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000, "%s", ex.what());
        return false;
    }
    sensor_to_base_footprint_.emplace(sensor_frame, extrinsic);
    sensor_frames_.push_back(sensor_frame);
    RCLCPP_INFO_STREAM(node_->get_logger(), "Lidar extrinsic " << base_frame_ << " <- " << sensor_frame
                                                                << ":\n"
                                                                << extrinsic.matrix());
    return true;
}

void LidarOdometryServer::InitializePoseAndExtrinsic(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg) {
    if (!tf2_buffer_->_frameExists(wheel_odom_frame_) || !tf2_buffer_->_frameExists(base_frame_) ||
        !tf2_buffer_->_frameExists(msg->header.frame_id)) {
        return;
    }

    // Independent from the service, start the ROS node from a known state.
    // ALICE M2: at the first scan's own stamp, which is where the first increment
    // starts. Upstream took the latest TF; the offline node buffers the bag 1 s
    // ahead, so the latest TF is ~1 s in the future and the whole trajectory came
    // out shifted by the distance travelled in that second (synthetic bag: 0.52 m).
    // Online it costs the latency times the speed when started while moving.
    Sophus::SE3d pose;
    try {
        pose = tf2::transformToSophus(tf2_buffer_->lookupTransform(
            wheel_odom_frame_, base_frame_, rclcpp::Time(msg->header.stamp),
            rclcpp::Duration(tf_timeout_)));
    } catch (tf2::TransformException &ex) {
        RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000,
                             "Waiting for %s -> %s at the first scan: %s", wheel_odom_frame_.c_str(),
                             base_frame_.c_str(), ex.what());
        return;
    }
    RCLCPP_INFO_STREAM(node_->get_logger(), "Resetting KISS-ICP pose:\n" << pose.matrix() << "\n");
    kinematic_icp_->SetPose(pose);

    Sophus::SE3d extrinsic;
    if (!LookupExtrinsic(msg->header.frame_id, extrinsic)) return;

    // Initialization finished
    RCLCPP_INFO(node_->get_logger(), "KISS-ICP ROS 2 odometry node initialized");
    timestamps_handler_.last_processed_stamp_ = msg->header.stamp;
    initialize_odom_node = true;
}

void LidarOdometryServer::RegisterFrame(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg) {
    if (!initialize_odom_node) {
        InitializePoseAndExtrinsic(msg);
        // ALICE M2: upstream carried on with an identity extrinsic when the TF was
        // not there yet; wait for it instead.
        if (!initialize_odom_node) return;
    }
    // ALICE M2: one extrinsic per sensor frame, so several lidars can feed one server.
    Sophus::SE3d extrinsic;
    if (!LookupExtrinsic(msg->header.frame_id, extrinsic)) return;

    // Buffer the last state This will be used for computing the veloicty
    const auto last_pose = kinematic_icp_->pose();
    const auto previous_stamp = timestamps_handler_.last_processed_stamp_;
    const auto &[begin_odom_query, end_odom_query, timestamps] =
        timestamps_handler_.ProcessTimestamps(msg);
    // ALICE M2: with two lidars a scan can end before the one processed last (their
    // latencies differ); its interval would run backwards. Drop it.
    if (rclcpp::Time(end_odom_query) <= rclcpp::Time(previous_stamp)) {
        timestamps_handler_.last_processed_stamp_ = previous_stamp;
        ++frames_skipped_out_of_order_;
        RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 5000,
                             "Dropped %zu out-of-order lidar frames so far (latest from %s)",
                             frames_skipped_out_of_order_, msg->header.frame_id.c_str());
        return;
    }

    // Get the initial guess from the wheel odometry
    const auto delta =
        LookupDeltaTransform(base_frame_, begin_odom_query, base_frame_, end_odom_query,
                             wheel_odom_frame_, tf_timeout_, tf2_buffer_);
    // ALICE M2: deskew over this scan's own span (first to last point). It equals
    // delta for one continuously spinning lidar; with two lidars interleaved the
    // span since the previous frame is shorter than the scan itself.
    const auto deskew_delta =
        LookupDeltaTransform(base_frame_, timestamps_handler_.last_scan_begin_stamp_, base_frame_,
                             end_odom_query, wheel_odom_frame_, tf_timeout_, tf2_buffer_);

    // Run kinematic ICP
    const Sophus::SE3d prior_pose = last_pose * delta;
    bool registered = false;
    if (delta.log().norm() > 1e-3) {
        const auto points = PointCloud2ToEigen(msg, {});
        const auto &[frame, kpoints] =
            kinematic_icp_->RegisterFrame(points, timestamps, extrinsic, delta, deskew_delta);
        PublishClouds(frame, kpoints);
        registered = true;
    } else {
        // ALICE M2: upstream dropped such small increments entirely, so a robot
        // creeping at < 1 mm per frame (two lidars: 20 frames/s) did not move at all
        // in odom_lidar. Carry the odometry increment; the map is not updated.
        kinematic_icp_->pose() = prior_pose;
    }

    // Compute velocities, use the elapsed time between the current msg and the last received
    const double elapsed_time =
        timestamps_handler_.toTime(end_odom_query) - timestamps_handler_.toTime(begin_odom_query);
    const Sophus::SE3d::Tangent delta_twist = (last_pose.inverse() * kinematic_icp_->pose()).log();
    const Sophus::SE3d::Tangent velocity =
        elapsed_time > 1e-4 ? Sophus::SE3d::Tangent(delta_twist / elapsed_time)
                            : Sophus::SE3d::Tangent::Zero();

    // ALICE M2: twist covariance from the registration information. The normalized
    // information H = J^T J / N + Omega gives the increment covariance
    // sigma^2 / N * H^-1 (sigma: point_sigma in prior mode, else the residual RMS),
    // scaled by covariance_scale since the points of one scan are not independent.
    const auto &diag = kinematic_icp_->LastRegistrationDiagnostics();
    if (covariance_from_registration_ && registered && diag.num_correspondences > 0 &&
        elapsed_time > 1e-4) {
        const double sigma2 = config_.odometry_prior ? config_.point_sigma * config_.point_sigma
                                                     : diag.mean_squared_residual;
        const Eigen::FullPivLU<Eigen::Matrix3d> lu(diag.information);
        if (lu.isInvertible()) {
            const Eigen::Matrix3d cov = covariance_scale_ * sigma2 /
                                        static_cast<double>(diag.num_correspondences) *
                                        lu.inverse() / (elapsed_time * elapsed_time);
            const int idx[3] = {0, 1, 5};  // x, y, yaw in the 6x6 twist covariance
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    odom_msg_.twist.covariance[idx[r] * 6 + idx[c]] = cov(r, c);
                }
            }
        }
    }

    // ALICE M2: registration diagnostics, one message per frame.
    //   0 stamp [s]  1 registered (0/1)  2 correspondences  3 of them point-to-line
    //   4 mean squared residual [m^2]  5..7 eigenvalues of the normalized information
    //   (ascending)  8..10 odometry term diagonal (x, y, yaw)  11..13 lidar correction
    //   of the prior (x [m], y [m], yaw [rad], robot frame)  14 frames dropped out of
    //   order so far  15 lidar index by frame (0 first seen, 1 second, ...)
    if (registration_debug_pub_->get_subscription_count() > 0) {
        std_msgs::msg::Float64MultiArray dbg;
        const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(diag.information);
        const Sophus::SE3d::Tangent corr = (prior_pose.inverse() * kinematic_icp_->pose()).log();
        const auto frame_it =
            std::find(sensor_frames_.cbegin(), sensor_frames_.cend(), msg->header.frame_id);
        dbg.data = {timestamps_handler_.toTime(end_odom_query),
                    registered ? 1.0 : 0.0,
                    static_cast<double>(diag.num_correspondences),
                    static_cast<double>(diag.num_point_to_line),
                    diag.mean_squared_residual,
                    es.eigenvalues()(0),
                    es.eigenvalues()(1),
                    es.eigenvalues()(2),
                    diag.omega_diagonal(0),
                    diag.omega_diagonal(1),
                    diag.omega_diagonal(2),
                    corr(0),
                    corr(1),
                    corr(5),
                    static_cast<double>(frames_skipped_out_of_order_),
                    static_cast<double>(std::distance(sensor_frames_.cbegin(), frame_it))};
        registration_debug_pub_->publish(dbg);
    }

    // Spit the current estimated pose to ROS pc_out_msgs handling the desired target frame
    PublishOdometryMsg(kinematic_icp_->pose(), velocity);
}

void LidarOdometryServer::PublishOdometryMsg(const Sophus::SE3d &pose,
                                             const Sophus::SE3d::Tangent &velocity) {
    // Broadcast over the tf tree
    if (publish_odom_tf_) {
        tf_msg_.transform = [&]() {
            if (invert_odom_tf_) return tf2::sophusToTransform(pose.inverse());
            return tf2::sophusToTransform(pose);
        }();
        tf_msg_.header.stamp = timestamps_handler_.last_processed_stamp_;
        tf_broadcaster_->sendTransform(tf_msg_);
    }

    // publish odometry msg
    odom_msg_.pose.pose = tf2::sophusToPose(pose);
    odom_msg_.twist.twist.linear.x = velocity[0];
    odom_msg_.twist.twist.linear.y = velocity[1];  // ALICE M2: holonomic base
    odom_msg_.twist.twist.angular.z = velocity[5];
    odom_msg_.header.stamp = timestamps_handler_.last_processed_stamp_;
    odom_publisher_->publish(odom_msg_);
}

void LidarOdometryServer::PublishClouds(const std::vector<Eigen::Vector3d> frame,
                                        const std::vector<Eigen::Vector3d> keypoints) {
    // For re-publishing the input frame and keypoints, we do it in the LiDAR coordinate frames
    std_msgs::msg::Header lidar_header;
    lidar_header.frame_id = base_frame_;
    lidar_header.stamp = timestamps_handler_.last_processed_stamp_;

    // The internal map representation is in the lidar_odom_frame_
    std_msgs::msg::Header map_header;
    map_header.frame_id = lidar_odom_frame_;
    map_header.stamp = timestamps_handler_.last_processed_stamp_;

    // Check for subscriptions before publishing to avoid unnecesary CPU usage
    if (frame_publisher_->get_subscription_count() > 0) {
        frame_publisher_->publish(std::move(EigenToPointCloud2(frame, lidar_header)));
    }
    if (kpoints_publisher_->get_subscription_count() > 0) {
        kpoints_publisher_->publish(std::move(EigenToPointCloud2(keypoints, lidar_header)));
    }
    if (map_publisher_->get_subscription_count() > 0) {
        map_publisher_->publish(
            std::move(EigenToPointCloud2(kinematic_icp_->LocalMap(), map_header)));
    }
}

}  // namespace kinematic_icp_ros
