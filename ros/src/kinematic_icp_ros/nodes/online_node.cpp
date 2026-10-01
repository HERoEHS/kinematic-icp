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
    for (const auto &topic : topics) {
        if (use_2d_lidar) {
            RCLCPP_INFO_STREAM(node_->get_logger(), "Started in 2D scanner mode with topic: " << topic);
            laser_scan_subs_.push_back(node_->create_subscription<sensor_msgs::msg::LaserScan>(
                topic, rclcpp::SensorDataQoS(),
                [&](const sensor_msgs::msg::LaserScan::ConstSharedPtr &msg) {
                    auto projected_scan = std::make_shared<sensor_msgs::msg::PointCloud2>();
                    laser_projector_.projectLaser(*msg, *projected_scan, -1.0,
                                                  laser_geometry::channel_option::Timestamp);
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

rclcpp::node_interfaces::NodeBaseInterface::SharedPtr OnlineNode::get_node_base_interface() {
    return node_->get_node_base_interface();
}

}  // namespace kinematic_icp_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(kinematic_icp_ros::OnlineNode)
