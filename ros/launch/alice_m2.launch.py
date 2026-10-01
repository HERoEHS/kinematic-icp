"""ALICE M2: Kinematic-ICP (holonomic) on both 2D lidars, live or from a bag.

  ros2 launch kinematic_icp alice_m2.launch.py
  ros2 launch kinematic_icp alice_m2.launch.py bag_filename:=/path/to/bag output_dir:=/tmp

Live, the node subscribes to the lidar topics and looks up odom -> base_footprint
from the running EKF. With bag_filename set, the offline node reads the lidar
topics and /tf, /tf_static from the bag and writes the poses in TUM format to
output_dir. Parameters: config/alice_m2.yaml (override with config_file:=).
"""
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _nodes(context):
    share = get_package_share_directory("kinematic_icp")
    config_file = LaunchConfiguration("config_file").perform(context) or share + "/config/alice_m2.yaml"
    bag = LaunchConfiguration("bag_filename").perform(context)
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context) == "true"
    common = {"use_sim_time": use_sim_time}
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
        DeclareLaunchArgument("use_sim_time", default_value="false", choices=["true", "false"]),
        OpaqueFunction(function=_nodes),
    ])
