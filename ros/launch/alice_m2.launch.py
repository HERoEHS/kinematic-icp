"""ALICE M2: Kinematic-ICP (holonomic) on both 2D lidars, live or from a bag.

  ros2 launch kinematic_icp alice_m2.launch.py
  ros2 launch kinematic_icp alice_m2.launch.py bag_filename:=/path/to/bag output_dir:=/tmp

Kinematic-ICP alone; for the EKF + correction setup use alice_m2_odometry.launch.py.
Live, the node subscribes to the lidar topics and looks up wheel_odom_frame ->
base_footprint from the running EKF. With bag_filename set, the offline node reads
the lidar topics and /tf, /tf_static from the bag and writes the poses in TUM
format to output_dir (no TF or correction published). Parameters:
config/alice_m2.yaml (override with config_file:=). wheel_odom_frame:=odom for
bags recorded with the EKF in its default frame.
"""
import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from m2_lidar_driver_params import lidar_blind_sectors  # noqa: E402


def _nodes(context):
    share = get_package_share_directory("kinematic_icp")
    config_file = LaunchConfiguration("config_file").perform(context) or share + "/config/alice_m2.yaml"
    bag = LaunchConfiguration("bag_filename").perform(context)
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context) == "true"
    common = {"use_sim_time": use_sim_time}
    # Blind sectors from the lidar driver's parameter files (see m2_lidar_driver_params).
    blind = lidar_blind_sectors()
    if blind:
        common["lidar_blind_sectors"] = blind
    wheel_odom_frame = LaunchConfiguration("wheel_odom_frame").perform(context)
    if wheel_odom_frame:
        common["wheel_odom_frame"] = wheel_odom_frame
    if bag:
        return [Node(
            package="kinematic_icp",
            executable="kinematic_icp_offline_node",
            name="offline_node",
            namespace="kinematic_icp",
            output="screen",
            parameters=[config_file, common, {
                "bag_filename": bag,
                "output_dir": LaunchConfiguration("output_dir").perform(context),
                "publish_odom_tf": False,
                "publish_correction_tf": False,
                "corrected_odometry_input_topic": "",
            }],
        )]
    return [Node(
        package="kinematic_icp",
        executable="kinematic_icp_online_node",
        name="online_node",
        namespace="kinematic_icp",
        output="screen",
        remappings=[("lidar_odometry", LaunchConfiguration("lidar_odometry_topic"))],
        parameters=[config_file, common],
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("config_file", default_value="",
                              description="Parameter file (default: config/alice_m2.yaml)"),
        DeclareLaunchArgument("bag_filename", default_value="",
                              description="Run the offline node on this bag instead of live"),
        DeclareLaunchArgument("output_dir", default_value=".",
                              description="Offline only: where the TUM pose file goes"),
        DeclareLaunchArgument("lidar_odometry_topic", default_value="lidar_odometry"),
        DeclareLaunchArgument("wheel_odom_frame", default_value="",
                              description="Override the prior frame (e.g. odom for older bags)"),
        DeclareLaunchArgument("use_sim_time", default_value="false", choices=["true", "false"]),
        OpaqueFunction(function=_nodes),
    ])
