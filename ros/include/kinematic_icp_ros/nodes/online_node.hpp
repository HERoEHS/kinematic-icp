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

#include <memory>
#include <string>
#include <vector>

// ROS
#include <laser_geometry/laser_geometry.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/parameter_client.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "kinematic_icp_ros/server/LidarOdometryServer.hpp"

namespace kinematic_icp_ros {

class OnlineNode {
public:
    OnlineNode() = delete;
    explicit OnlineNode(const rclcpp::NodeOptions &options);

    // Neccesary for ROS 2 composition
    rclcpp::node_interfaces::NodeBaseInterface::SharedPtr get_node_base_interface();

private:
    // Common for offline/online nodes
    std::string lidar_topic_;
    std::shared_ptr<LidarOdometryServer> odometry_server_;
    rclcpp::Node::SharedPtr node_;
    laser_geometry::LaserProjection laser_projector_;

    // Online node specifics
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr laser_scan_sub_;
    // ALICE M2: one subscription per entry of lidar_topics (several 2D lidars).
    std::vector<rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr> laser_scan_subs_;
    std::vector<rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr> pointcloud_subs_;
    // ALICE M2: beam time from angle (utils::RetimeLaserCloudFromAngle).
    bool laser_time_from_angle_{false};
    double laser_time_origin_rad_{0.0};
    bool laser_time_increasing_{true};

    // ALICE M2: ask the driver publishing each lidar topic for its ignore_array (the
    // sectors it blanks), so moving_object_filter uses what the driver really does.
    struct DriverQuery {
        std::string topic;
        std::string driver;
        std::shared_ptr<rclcpp::AsyncParametersClient> client;
        int tries{0};
        bool done{false};
    };
    void QueryDriverBlindSectors(std::size_t index, const std::string &frame_id);
    std::vector<DriverQuery> driver_queries_;
};

}  // namespace kinematic_icp_ros
