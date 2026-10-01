"""ALICE M2 odometry: wheel + gyro EKF, optionally corrected by both 2D lidars.

  ros2 launch kinematic_icp alice_m2_odometry.launch.py                        # lidar_correction:=true
  ros2 launch kinematic_icp alice_m2_odometry.launch.py lidar_correction:=false

lidar_correction:=false starts the EKF exactly as robot_localization's
wio_ekf.launch.py does (same node name, same shared wio_ekf.yaml): /odometry/filtered
and odom -> base_footprint, the same as on robots without lidars.

lidar_correction:=true keeps the shared files untouched and only renames at launch:
  EKF            frame odom_ekf, topic /odometry/ekf, TF odom_ekf -> base_footprint
  Kinematic-ICP  TF odom -> odom_ekf (lidar correction), and /odometry/filtered in
                 frame odom = EKF odometry with the correction applied
so SLAM, Nav2, docking and the health monitor keep using /odometry/filtered and odom.
Kinematic-ICP parameters: config/alice_m2.yaml (kicp_config:= to override).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _nodes(context):
    rl_share = get_package_share_directory("robot_localization")
    kicp_share = get_package_share_directory("kinematic_icp")
    ekf_params = LaunchConfiguration("ekf_params").perform(context) or os.path.join(
        rl_share, "params", "wio_ekf.yaml")
    kicp_config = LaunchConfiguration("kicp_config").perform(context) or os.path.join(
        kicp_share, "config", "alice_m2.yaml")
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context) == "true"
    correction = LaunchConfiguration("lidar_correction").perform(context) == "true"

    if not correction:
        return [Node(
            package="robot_localization", executable="ekf_node", name="ekf_filter_node",
            output="screen", parameters=[ekf_params, {"use_sim_time": use_sim_time}],
        )]

    ekf_frame, frame = "odom_ekf", "odom"
    ekf_topic, topic = "/odometry/ekf", "/odometry/filtered"
    return [
        Node(
            package="robot_localization", executable="ekf_node", name="ekf_filter_node",
            output="screen",
            parameters=[ekf_params, {"use_sim_time": use_sim_time,
                                     "odom_frame": ekf_frame, "world_frame": ekf_frame}],
            remappings=[("odometry/filtered", ekf_topic)],
        ),
        Node(
            package="kinematic_icp", executable="kinematic_icp_online_node", name="online_node",
            namespace="kinematic_icp", output="screen",
            parameters=[kicp_config, {
                "use_sim_time": use_sim_time,
                "wheel_odom_frame": ekf_frame,
                "lidar_odom_frame": frame,
                "publish_odom_tf": False,
                "publish_correction_tf": True,
                "corrected_odometry_input_topic": ekf_topic,
                "corrected_odometry_output_topic": topic,
            }],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("lidar_correction", default_value="true", choices=["true", "false"]),
        DeclareLaunchArgument("ekf_params", default_value="",
                              description="EKF parameters (default: robot_localization wio_ekf.yaml)"),
        DeclareLaunchArgument("kicp_config", default_value="",
                              description="Kinematic-ICP parameters (default: config/alice_m2.yaml)"),
        DeclareLaunchArgument("use_sim_time", default_value="false", choices=["true", "false"]),
        OpaqueFunction(function=_nodes),
    ])
