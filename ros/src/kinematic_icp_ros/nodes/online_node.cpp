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
#include <memory>
#include <rclcpp/logging.hpp>
#include <string>

// ROS
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <laser_geometry/laser_geometry.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/node_options.hpp>
#include <rclcpp/subscription.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "kinematic_icp_ros/nodes/online_node.hpp"
#include "kinematic_icp_ros/utils/RosUtils.hpp"
#include "kinematic_icp_ros/server/LidarOdometryServer.hpp"

namespace kinematic_icp_ros {

OnlineNode ::OnlineNode(const rclcpp::NodeOptions &options) {
    node_ = rclcpp::Node::make_shared("kinematic_icp_online_node", options);
    // ALICE M2: lidar_topics (several sensors into one estimate) takes precedence
    // over the single upstream lidar_topic. Each frame is registered with the
    // extrinsic of its own header.frame_id.
    lidar_topic_ = node_->declare_parameter<std::string>("lidar_topic", "");
    auto topics = node_->declare_parameter<std::vector<std::string>>("lidar_topics",
                                                                     std::vector<std::string>{});
    topics.erase(std::remove(topics.begin(), topics.end(), std::string{}), topics.end());
    if (topics.empty() && !lidar_topic_.empty()) topics.push_back(lidar_topic_);
    if (topics.empty()) {
        throw std::runtime_error("kinematic_icp: set lidar_topic or lidar_topics");
    }
    odometry_server_ = std::make_shared<LidarOdometryServer>(node_);
    const bool use_2d_lidar = node_->declare_parameter<bool>("use_2d_lidar");
    laser_time_from_angle_ = node_->declare_parameter<bool>("laser_time_from_angle", false);
    laser_time_origin_rad_ =
        node_->declare_parameter<double>("laser_time_origin_deg", 0.0) * M_PI / 180.0;
    laser_time_increasing_ = node_->declare_parameter<bool>("laser_time_increasing", true);
    const bool blind_from_driver =
        node_->declare_parameter<bool>("lidar_blind_sectors_from_driver", true);
    for (const auto &topic : topics) driver_queries_.push_back({topic, "", nullptr, 0, !blind_from_driver});
    for (std::size_t index = 0; index < topics.size(); ++index) {
        const auto &topic = topics[index];
        if (use_2d_lidar) {
            RCLCPP_INFO_STREAM(node_->get_logger(), "Started in 2D scanner mode with topic: " << topic);
            laser_scan_subs_.push_back(node_->create_subscription<sensor_msgs::msg::LaserScan>(
                topic, rclcpp::SensorDataQoS(),
                [this, index](const sensor_msgs::msg::LaserScan::ConstSharedPtr &msg) {
                    QueryDriverBlindSectors(index, msg->header.frame_id);
                    auto projected_scan = std::make_shared<sensor_msgs::msg::PointCloud2>();
                    laser_projector_.projectLaser(
                        *msg, *projected_scan, -1.0,
                        laser_geometry::channel_option::Timestamp | laser_geometry::channel_option::Index);
                    if (laser_time_from_angle_) {
                        utils::RetimeLaserCloudFromAngle(*projected_scan, *msg, laser_time_origin_rad_,
                                                         laser_time_increasing_);
                    }
                    odometry_server_->RegisterFrame(projected_scan);
                }));
        } else {
            RCLCPP_INFO_STREAM(node_->get_logger(), "Started in 3D Lidar mode with topic: " << topic);
            pointcloud_subs_.push_back(node_->create_subscription<sensor_msgs::msg::PointCloud2>(
                topic, rclcpp::SensorDataQoS(),
                [&](const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg) {
                    odometry_server_->RegisterFrame(msg);
                }));
        }
    }
}

void OnlineNode::QueryDriverBlindSectors(const std::size_t index, const std::string &frame_id) {
    auto &query = driver_queries_[index];
    if (query.done) return;
    if (!query.client) {
        // Right after start-up discovery may know the publisher but not its node name yet.
        const auto publishers = node_->get_publishers_info_by_topic(query.topic);
        if (!publishers.empty() &&
            publishers.front().node_name().find("_UNKNOWN_") == std::string::npos) {
            const auto &ns = publishers.front().node_namespace();
            query.driver = (ns == "/" ? std::string{} : ns) + "/" + publishers.front().node_name();
            query.client = std::make_shared<rclcpp::AsyncParametersClient>(node_, query.driver);
        }
    }
    if (!query.client || !query.client->service_is_ready()) {
        if (++query.tries > 100) {  // ~10 s of scans
            query.done = true;
            RCLCPP_WARN(node_->get_logger(),
                        "%s: could not reach the driver (%s), keeping lidar_blind_sectors",
                        query.topic.c_str(), query.driver.empty() ? "node unknown" : query.driver.c_str());
        }
        return;
    }
    query.done = true;
    query.client->get_parameters(
        {"ignore_array"}, [this, frame_id, driver = query.driver](
                              std::shared_future<std::vector<rclcpp::Parameter>> future) {
            const auto params = future.get();
            if (params.empty() || params.front().get_type() != rclcpp::ParameterType::PARAMETER_STRING) {
                RCLCPP_INFO(node_->get_logger(),
                            "%s has no ignore_array, keeping lidar_blind_sectors for %s",
                            driver.c_str(), frame_id.c_str());
                return;
            }
            odometry_server_->SetLidarBlindSectors(
                frame_id, utils::ParseAngleSectors(params.front().as_string()), "driver " + driver);
        });
}

rclcpp::node_interfaces::NodeBaseInterface::SharedPtr OnlineNode::get_node_base_interface() {
    return node_->get_node_base_interface();
}

}  // namespace kinematic_icp_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(kinematic_icp_ros::OnlineNode)
