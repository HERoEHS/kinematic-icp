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
#include <cmath>
#include <cstdio>
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
// ALICE M2: below these the robot stands (RegisterFrame).
constexpr double kStandingSpeed = 0.005;              // [m/s]
constexpr double kStandingRate = 0.3 * M_PI / 180.0;  // [rad/s]
// ALICE M2: a turn of the gyro alone counts as no turn once kept up this long, gaps up to
// kGyroOnlyTurnGap allowed (RegisterFrame, standing_odometry_topic).
constexpr double kGyroOnlyTurnTime = 0.5;  // [s]
constexpr double kGyroOnlyTurnGap = 0.5;   // [s]
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
    // prior_sigma_xy_rel (older configs) sets both axes unless they are given.
    const double prior_sigma_xy_rel =
        node->declare_parameter<double>("prior_sigma_xy_rel", config.prior_sigma_x_rel);
    config.prior_sigma_x_rel = node->declare_parameter<double>("prior_sigma_x_rel", prior_sigma_xy_rel);
    config.prior_sigma_y_rel = node->declare_parameter<double>("prior_sigma_y_rel", prior_sigma_xy_rel);
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
    config.moving_object_filter =
        node->declare_parameter<bool>("moving_object_filter", config.moving_object_filter);
    config.moving_object_min_motion =
        node->declare_parameter<double>("moving_object_min_motion", config.moving_object_min_motion);
    config.moving_object_same_tolerance = node->declare_parameter<double>(
        "moving_object_same_tolerance", config.moving_object_same_tolerance);
    config.moving_object_static_tolerance = node->declare_parameter<double>(
        "moving_object_static_tolerance", config.moving_object_static_tolerance);
    config.moving_object_history =
        node->declare_parameter<int>("moving_object_history", config.moving_object_history);
    config.moving_object_footprint = node->declare_parameter<std::vector<double>>(
        "moving_object_footprint", config.moving_object_footprint);
    // Blind sectors per lidar frame, entries "frame_id:from,to[,from,to...]" in degrees
    // (the driver's ignore_array). The launch files fill it from the driver's
    // parameter files; the online node then asks the running driver.
    for (const auto &entry : node->declare_parameter<std::vector<std::string>>(
             "lidar_blind_sectors", std::vector<std::string>{})) {
        const auto colon = entry.find(':');
        if (colon == std::string::npos) {
            RCLCPP_WARN(node_->get_logger(), "lidar_blind_sectors: '%s' is not frame_id:from,to",
                        entry.c_str());
            continue;
        }
        SetLidarBlindSectors(entry.substr(0, colon),
                             utils::ParseAngleSectors(entry.substr(colon + 1)),
                             "parameter lidar_blind_sectors");
    }
    covariance_from_registration_ =
        node->declare_parameter<bool>("covariance_from_registration", covariance_from_registration_);
    covariance_scale_ = node->declare_parameter<double>("covariance_scale", covariance_scale_);
    kinematic_icp::DegeneracyOptions degeneracy;
    degeneracy.window = node->declare_parameter<double>("degeneracy_window", degeneracy.window);
    degeneracy.ratio_enter =
        node->declare_parameter<double>("degeneracy_ratio_enter", degeneracy.ratio_enter);
    degeneracy.ratio_exit =
        node->declare_parameter<double>("degeneracy_ratio_exit", degeneracy.ratio_exit);
    degeneracy.min_matches_enter = node->declare_parameter<double>(
        "degeneracy_min_matches_enter", degeneracy.min_matches_enter);
    degeneracy.min_matches_exit = node->declare_parameter<double>(
        "degeneracy_min_matches_exit", degeneracy.min_matches_exit);
    degeneracy.min_frames = static_cast<std::size_t>(std::max<int64_t>(
        1, node->declare_parameter<int64_t>("degeneracy_min_frames",
                                            static_cast<int64_t>(degeneracy.min_frames))));
    degeneracy_ = std::make_unique<kinematic_icp::LidarDegeneracy>(degeneracy);
    publish_correction_tf_ =
        node->declare_parameter<bool>("publish_correction_tf", publish_correction_tf_);
    correction_tf_rate_ = node->declare_parameter<double>("correction_tf_rate", correction_tf_rate_);
    correction_tf_post_date_ =
        node->declare_parameter<double>("correction_tf_post_date", correction_tf_post_date_);
    corrected_odometry_input_topic_ = node->declare_parameter<std::string>(
        "corrected_odometry_input_topic", corrected_odometry_input_topic_);
    corrected_odometry_output_topic_ = node->declare_parameter<std::string>(
        "corrected_odometry_output_topic", corrected_odometry_output_topic_);
    standing_odometry_topic_ =
        node->declare_parameter<std::string>("standing_odometry_topic", standing_odometry_topic_);
    if (publish_correction_tf_ && publish_odom_tf_) {
        // base -> lidar_odom_frame plus lidar_odom_frame -> wheel_odom_frame would close
        // a loop in the TF tree.
        RCLCPP_WARN(node_->get_logger(),
                    "publish_correction_tf is set: not publishing the %s pose TF as well",
                    lidar_odom_frame_.c_str());
        publish_odom_tf_ = false;
    }
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
    // ALICE M2: lidar degeneracy, see UpdateDegeneracy(). Published every frame (no
    // subscriber check) so a consumer can tell a silent odometry from a quiet scene.
    degeneracy_pub_ =
        node_->create_publisher<diagnostic_msgs::msg::DiagnosticArray>("degeneracy", qos);

    set_pose_srv_ = node_->create_service<std_srvs::srv::Trigger>(
        "set_pose", [&](const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                        std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
            // ALICE M2: in correction mode keep the current correction, so the
            // published lidar_odom_frame does not jump back to the EKF.
            const auto pose =
                correction_ * LookupTransform(wheel_odom_frame_, base_frame_, tf2_buffer_);
            RCLCPP_WARN_STREAM(node_->get_logger(), "Resetting KISS-ICP pose:\n"
                                                        << pose.matrix() << "\n");
            kinematic_icp_->SetPose(pose);
            response->success = true;
        });

    // Initialize the transform broadcaster
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(node_);
    tf2_buffer_ = std::make_unique<tf2_ros::Buffer>(node_->get_clock());
    tf2_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf2_buffer_);

    // ALICE M2 correction mode (identity until the first registration).
    if (publish_correction_tf_ && correction_tf_rate_ > 0.0) {
        correction_timer_ = node_->create_wall_timer(
            std::chrono::duration<double>(1.0 / correction_tf_rate_), [this]() { PublishCorrectionTf(); });
    }
    if (!corrected_odometry_input_topic_.empty() && !corrected_odometry_output_topic_.empty()) {
        corrected_odometry_pub_ =
            node_->create_publisher<nav_msgs::msg::Odometry>(corrected_odometry_output_topic_, 10);
        corrected_odometry_sub_ = node_->create_subscription<nav_msgs::msg::Odometry>(
            corrected_odometry_input_topic_, 10,
            [this](const nav_msgs::msg::Odometry::ConstSharedPtr msg) { RepublishCorrectedOdometry(msg); });
        RCLCPP_INFO(node_->get_logger(), "Republishing %s as %s in frame %s",
                    corrected_odometry_input_topic_.c_str(), corrected_odometry_output_topic_.c_str(),
                    lidar_odom_frame_.c_str());
    }
    if (!standing_odometry_topic_.empty()) {
        RCLCPP_INFO(node_->get_logger(),
                    "Standing: a gyro turn the wheels (%s) do not show, kept up %.1f s, is no turn",
                    standing_odometry_topic_.c_str(), kGyroOnlyTurnTime);
    }

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

void LidarOdometryServer::SetLidarBlindSectors(const std::string &frame_id,
                                               const std::vector<double> &sectors_deg,
                                               const std::string &source) {
    std::ostringstream text;
    for (std::size_t k = 0; k + 1 < sectors_deg.size(); k += 2) {
        text << (k ? ", " : "") << sectors_deg[k] << ".." << sectors_deg[k + 1];
    }
    RCLCPP_INFO(node_->get_logger(), "Blind sectors of %s from %s: %s deg", frame_id.c_str(),
                source.c_str(), sectors_deg.empty() ? "none" : text.str().c_str());
    blind_sectors_by_frame_[frame_id] = sectors_deg;
    blind_sectors_applied_.erase(frame_id);  // passed on with the next frame of that lidar
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
    const int sensor = static_cast<int>(std::distance(
        sensor_frames_.cbegin(),
        std::find(sensor_frames_.cbegin(), sensor_frames_.cend(), msg->header.frame_id)));
    if (last_sensor_ < 0) last_sensor_ = sensor;
    // ALICE M2: the frame processed last came from the other lidar.
    const bool other_lidar = sensor != last_sensor_;

    // Buffer the last state This will be used for computing the veloicty
    const auto last_pose = kinematic_icp_->pose();
    const auto previous_stamp = timestamps_handler_.last_processed_stamp_;
    const auto &[begin_odom_query, end_odom_query, timestamps] =
        timestamps_handler_.ProcessTimestamps(msg);
    // ALICE M2: with two lidars a scan can end before the one processed last (their
    // latencies differ); its interval would run backwards. Drop it. A scan of the other
    // lidar ending at the same instant is kept: it sees the same moment from another
    // place (simulated lidars spin in lockstep, so this is every second frame there).
    const rclcpp::Time end_time(end_odom_query), previous_time(previous_stamp);
    if (end_time < previous_time || (end_time == previous_time && !other_lidar)) {
        timestamps_handler_.last_processed_stamp_ = previous_stamp;
        ++frames_skipped_out_of_order_;
        RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 5000,
                             "Dropped %zu out-of-order lidar frames so far (latest from %s)",
                             frames_skipped_out_of_order_, msg->header.frame_id.c_str());
        return;
    }
    last_sensor_ = sensor;

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
    if (blind_sectors_applied_.insert(msg->header.frame_id).second) {
        const auto it = blind_sectors_by_frame_.find(msg->header.frame_id);
        if (it != blind_sectors_by_frame_.cend()) {
            kinematic_icp_->SetBlindSectors(sensor, it->second);
        } else if (config_.moving_object_filter) {
            RCLCPP_WARN(node_->get_logger(),
                        "No blind sectors for %s: moving_object_filter judges all directions",
                        msg->header.frame_id.c_str());
        }
    }
    // ALICE M2: whether the robot moves is judged on the increment since the frame
    // processed last. A scan of the other lidar ending at the same instant (lockstep
    // simulated lidars, every second frame) has none although the robot moved; the
    // motion during this scan decides then. Not for scans merely close in time (a few
    // ms apart now and then on the robot): registering those made the robot bags worse
    // (2026-10-19, tape-measured end error over six runs 4.7 -> 6.9 cm), so the robot
    // keeps skipping them as before.
    const bool moved_since_last = delta.log().norm() > 1e-3;
    const bool same_instant = other_lidar && end_time == previous_time;
    const bool moved_in_scan = same_instant && deskew_delta.log().norm() > 1e-3;
    // ALICE M2: increments below the threshold used to be carried for good, so whatever
    // the odometry drifted by while the robot stood (gyro bias walk, slow slip) was never
    // checked against the scans: standing 7 min in Isaac took the pose 2.5 deg and 0.4 m
    // off with the correction frozen (2026-10-26). Standing (below 5 mm/s and 0.3 deg/s
    // since the frame processed last), sum the carried increments and register once the
    // sum passes the threshold. Only standing: summing slow motion too (starting,
    // stopping, slow turns) moved the robot bags by up to 6 cm and 2.9 deg and the
    // tape-measured end errors got worse on four of six. Slow motion is carried as before
    // and empties the sum. Judged on speed, not on the increment: the simulated EKF
    // jitters ~0.6 mm/s standing and its lidar frames are 100 ms apart, so 1e-4 per frame
    // emptied the sum every few frames. A frame of the other lidar a few ms after the last
    // one says nothing about speed and leaves the sum. The carried increments are already
    // in the pose, so the registration still gets this frame's increment; the scans then
    // take out what the carried odometry got wrong. Such a frame is not added to the local
    // map (KinematicICP::RegisterFrame, update_map).
    const double frame_dt =
        timestamps_handler_.toTime(end_odom_query) - timestamps_handler_.toTime(begin_odom_query);
    const Sophus::SE3d::Tangent step = delta.log();
    const bool too_close = same_instant || frame_dt < 0.03;
    // ALICE M2: with standing_odometry_topic a turn the prior shows but the wheels do not,
    // kept up for kGyroOnlyTurnTime, is no turn. The prior's heading is the gyro alone, and
    // on the robot (2026-10-28) the gyro read about +0.5 deg/s for 3 to 7 s at a time while
    // the base did not turn (lidar and wheels both 0 deg/s): judged on the prior that was
    // slow motion, carried without a scan, and the heading went 11 deg off. Only the turn:
    // the wheels' twist judging standing outright called standing frames moving on brief
    // wheel spikes while steering (78 to 202 frames a bag). Only kept up: the gyro alone
    // also turns for a frame or five when steering starts (crab_01, 0.8 to 1.6 deg/s, likely
    // the base twisting), and registering those frames in the corridor cost 0.5 deg.
    const double begin_time = timestamps_handler_.toTime(begin_odom_query);
    const double end_time_s = timestamps_handler_.toTime(end_odom_query);
    const bool prior_moves = step.head<3>().norm() >= kStandingSpeed * frame_dt;
    const bool prior_turns = step.tail<3>().norm() >= kStandingRate * frame_dt;
    const int wheels_turn =
        too_close || prior_moves || !prior_turns ? -1 : WheelsTurning(begin_time, end_time_s);
    const bool gyro_only_turn = wheels_turn == 0;
    if (gyro_only_turn) {
        // Frames in between that move, turn on the wheels too or turn below the rate do not
        // end it (the robot drift had a few of each every second); kGyroOnlyTurnGap without a
        // gyro-only turn starts it over.
        if (gyro_only_turn_since_ < 0.0 || begin_time - gyro_only_turn_last_ > kGyroOnlyTurnGap) {
            gyro_only_turn_since_ = begin_time;
        }
        gyro_only_turn_last_ = end_time_s;
    }
    const bool kept_up =
        gyro_only_turn && end_time_s - gyro_only_turn_since_ >= kGyroOnlyTurnTime;
    const bool standing = !too_close && !prior_moves && (!prior_turns || kept_up);
    const bool accumulated = standing && !moved_since_last &&
                             (unregistered_motion_ * delta).log().norm() > 1e-3;
    if (moved_since_last || moved_in_scan || accumulated) {
        const auto points = PointCloud2ToEigen(msg, {});
        // Standing, the frame only corrects the pose; the local map is left as it is.
        const auto &[frame, kpoints] = kinematic_icp_->RegisterFrame(
            points, timestamps, extrinsic, delta, deskew_delta, sensor, !accumulated);
        PublishClouds(frame, kpoints);
        registered = true;
        unregistered_motion_ = Sophus::SE3d();
    } else {
        // ALICE M2: upstream dropped such small increments entirely, so a robot
        // creeping at < 1 mm per frame (two lidars: 20 frames/s) did not move at all
        // in odom_lidar. Carry the odometry increment; the map is not updated.
        kinematic_icp_->pose() = prior_pose;
        // Standing adds to the sum, slow motion empties it, a frame too close to the last
        // one (the other lidar) leaves it.
        if (standing) {
            unregistered_motion_ = unregistered_motion_ * delta;
        } else if (!too_close) {
            unregistered_motion_ = Sophus::SE3d();
        }
    }

    // Compute velocities, use the elapsed time between the current msg and the last received
    const double elapsed_time = frame_dt;
    const Sophus::SE3d::Tangent delta_twist = (last_pose.inverse() * kinematic_icp_->pose()).log();
    // ALICE M2: no time since the frame before (the other lidar at the same instant), so
    // no interval to divide by: keep the last velocity.
    const Sophus::SE3d::Tangent velocity =
        same_instant          ? last_velocity_
        : elapsed_time > 1e-4 ? Sophus::SE3d::Tangent(delta_twist / elapsed_time)
                              : Sophus::SE3d::Tangent::Zero();
    last_velocity_ = velocity;

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
    //   order so far  15 lidar index by frame (0 first seen, 1 second, ...)  16 points
    //   left out as moving with the robot (moving_object_filter)  17 the other lidar
    //   ended at the same instant just before (0/1; registered when moving, see above)
    //   18 registered only because the carried increments added up (0/1, see above)
    //   19 the wheels asked about a turn of the prior: 1 they turn, 0 they do not (standing
    //   once kept up, kGyroOnlyTurnTime), -1 not asked (no turn, moving, no wheel sample or
    //   no standing_odometry_topic)
    if (registration_debug_pub_->get_subscription_count() > 0) {
        std_msgs::msg::Float64MultiArray dbg;
        const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(diag.information);
        const Sophus::SE3d::Tangent corr = (prior_pose.inverse() * kinematic_icp_->pose()).log();
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
                    static_cast<double>(sensor),
                    registered ? static_cast<double>(kinematic_icp_->LastNumMovingPoints()) : 0.0,
                    same_instant ? 1.0 : 0.0,
                    accumulated ? 1.0 : 0.0,
                    static_cast<double>(wheels_turn)};
        registration_debug_pub_->publish(dbg);
    }

    // ALICE M2: correction mode
    if (publish_correction_tf_ || corrected_odometry_pub_) UpdateCorrection(end_odom_query);

    // ALICE M2: lidar degeneracy over the last frames. A frame registered only because the
    // carried increments added up is left out, like the skipped ones before: standing, the
    // window keeps describing the last place instead of one lone frame (not judged).
    UpdateDegeneracy(end_odom_query, registered && !accumulated);

    // Spit the current estimated pose to ROS pc_out_msgs handling the desired target frame
    PublishOdometryMsg(kinematic_icp_->pose(), velocity);
}

void LidarOdometryServer::UpdateCorrection(const builtin_interfaces::msg::Time &stamp) {
    // correction = (lidar pose of base) * (EKF pose of base)^-1 at the same instant.
    try {
        const Sophus::SE3d wheel_pose = tf2::transformToSophus(tf2_buffer_->lookupTransform(
            wheel_odom_frame_, base_frame_, rclcpp::Time(stamp), rclcpp::Duration(tf_timeout_)));
        correction_ = kinematic_icp_->pose() * wheel_pose.inverse();
    } catch (tf2::TransformException &ex) {
        RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000,
                             "Correction kept (no %s -> %s at the scan end): %s",
                             wheel_odom_frame_.c_str(), base_frame_.c_str(), ex.what());
    }
}

void LidarOdometryServer::PublishCorrectionTf() {
    geometry_msgs::msg::TransformStamped t;
    // Post-dated like a localizer's map -> odom, so lookups at "now" never wait.
    t.header.stamp = node_->now() + rclcpp::Duration::from_seconds(correction_tf_post_date_);
    t.header.frame_id = lidar_odom_frame_;
    t.child_frame_id = wheel_odom_frame_;
    t.transform = tf2::sophusToTransform(correction_);
    tf_broadcaster_->sendTransform(t);
}

void LidarOdometryServer::RepublishCorrectedOdometry(
    const nav_msgs::msg::Odometry::ConstSharedPtr &msg) {
    if (msg->header.frame_id != wheel_odom_frame_) {
        RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 5000,
                             "%s is in frame %s, expected %s", corrected_odometry_input_topic_.c_str(),
                             msg->header.frame_id.c_str(), wheel_odom_frame_.c_str());
    }
    nav_msgs::msg::Odometry out = *msg;
    out.header.frame_id = lidar_odom_frame_;
    out.pose.pose = tf2::sophusToPose(correction_ * tf2::poseToSophus(msg->pose.pose));
    // Pose covariance rotated into lidar_odom_frame; the twist is in the child frame
    // and does not change.
    const Eigen::Matrix3d R = correction_.so3().matrix();
    Eigen::Matrix<double, 6, 6> J = Eigen::Matrix<double, 6, 6>::Zero();
    J.block<3, 3>(0, 0) = R;
    J.block<3, 3>(3, 3) = R;
    const Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> C(msg->pose.covariance.data());
    const Eigen::Matrix<double, 6, 6, Eigen::RowMajor> Co = J * C * J.transpose();
    Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(out.pose.covariance.data()) = Co;
    corrected_odometry_pub_->publish(out);
}

void LidarOdometryServer::PublishOdometryMsg(const Sophus::SE3d &pose,
                                             const Sophus::SE3d::Tangent &velocity) {
    // Broadcast over the tf tree
    // ALICE M2: a frame of the other lidar ending at the same instant must not repeat
    // the TF stamp (tf2 rejects it as TF_REPEATED_DATA).
    if (publish_odom_tf_ &&
        rclcpp::Time(timestamps_handler_.last_processed_stamp_) != rclcpp::Time(tf_msg_.header.stamp)) {
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

void LidarOdometryServer::UpdateDegeneracy(const builtin_interfaces::msg::Time &stamp,
                                           const bool registered) {
    // Only the holonomic registration fills the diagnostics. A frame skipped because
    // the robot stood still adds nothing: the window keeps describing the last place.
    if (registered && config_.holonomic) {
        const auto &diag = kinematic_icp_->LastRegistrationDiagnostics();
        // Scan-only part: the odometry term Omega is diagonal.
        Eigen::Matrix3d per_match = diag.information;
        per_match.diagonal() -= diag.omega_diagonal;
        const double sigma = config_.odometry_prior ? config_.point_sigma
                                                    : std::sqrt(diag.mean_squared_residual);
        const Eigen::Matrix3d R = kinematic_icp_->pose().so3().matrix();
        degeneracy_->AddFrame(timestamps_handler_.toTime(stamp), std::atan2(R(1, 0), R(0, 0)),
                              per_match, diag.num_correspondences, sigma);
    }
    const auto &state = degeneracy_->state();

    diagnostic_msgs::msg::DiagnosticArray msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = lidar_odom_frame_;
    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name = "kinematic_icp: lidar degeneracy";
    status.hardware_id = lidar_odom_frame_;  // weak_axis_deg is in this frame
    char text[160];
    if (degeneracy_->Empty()) {
        status.level = diagnostic_msgs::msg::DiagnosticStatus::STALE;
        status.message = "no registered frame yet";
    } else if (!state.judged) {
        status.level = diagnostic_msgs::msg::DiagnosticStatus::STALE;
        std::snprintf(text, sizeof(text), "too few registered frames to judge (%zu)", state.frames);
        status.message = text;
    } else if (!state.degenerate) {
        status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
        status.message = "lidar fixes the position";
    } else if (state.reason == kinematic_icp::DegeneracyReason::kSparse) {
        status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
        std::snprintf(text, sizeof(text), "sparse: %.1f matches per frame, position from odometry only",
                      state.matches_per_frame);
        status.message = text;
    } else {
        status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
        std::snprintf(text, sizeof(text), "corridor: position along %.0f deg from odometry only",
                      degeneracy_->WeakAxisDegrees());
        status.message = text;
    }
    auto value = [&status](const std::string &key, const std::string &v) {
        diagnostic_msgs::msg::KeyValue kv;
        kv.key = key;
        kv.value = v;
        status.values.push_back(kv);
    };
    auto number = [&value](const std::string &key, const double v) {
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), "%.6g", v);
        value(key, buffer);
    };
    value("degenerate", state.degenerate ? "true" : "false");
    value("reason", kinematic_icp::ToString(state.reason));
    number("ratio", state.ratio);
    number("weak_axis_deg", degeneracy_->WeakAxisDegrees());
    number("sigma_weak", state.sigma_weak);
    number("sigma_strong", state.sigma_strong);
    number("sigma_yaw", state.sigma_yaw);
    number("matches_per_frame", state.matches_per_frame);
    number("frames", static_cast<double>(state.frames));
    msg.status.push_back(status);
    degeneracy_pub_->publish(msg);

    if (state.judged && state.degenerate != degeneracy_reported_) {
        degeneracy_reported_ = state.degenerate;
        if (state.degenerate) {
            RCLCPP_INFO(node_->get_logger(), "Lidar degenerate (%s, ratio %.3f, %.1f matches per frame)",
                        status.message.c_str(), state.ratio, state.matches_per_frame);
        } else {
            RCLCPP_INFO(node_->get_logger(), "Lidar fixes the position again (ratio %.3f)",
                        state.ratio);
        }
    }
}

void LidarOdometryServer::AddStandingOdometry(const nav_msgs::msg::Odometry &msg) {
    const double stamp = rclcpp::Time(msg.header.stamp).seconds();
    standing_rates_.emplace_back(stamp, std::abs(msg.twist.twist.angular.z));
    while (standing_rates_.front().first < stamp - 2.0) standing_rates_.pop_front();
}

int LidarOdometryServer::WheelsTurning(const double begin, const double end) const {
    if (standing_odometry_topic_.empty()) return -1;
    // The samples over the frame and a wheel period beyond either end (50 Hz on the robot);
    // any of them turning makes the frame turning.
    constexpr double margin = 0.03;  // [s]
    bool sampled = false;
    for (const auto &[stamp, rate] : standing_rates_) {
        if (stamp < begin - margin || stamp > end + margin) continue;
        if (rate >= kStandingRate) return 1;
        sampled = true;
    }
    return sampled ? 0 : -1;
}

}  // namespace kinematic_icp_ros
