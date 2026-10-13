"""ALICE M2 odometry: wheel + gyro EKF, optionally corrected by both 2D lidars.

  ros2 launch kinematic_icp alice_m2_odometry.launch.py                        # lidar_correction:=true
  ros2 launch kinematic_icp alice_m2_odometry.launch.py lidar_correction:=false

lidar_correction:=false starts the EKF exactly as robot_localization's
wio_ekf.launch.py does in an ALICE_GENERATION=alice_m2 shell (same node name, same
shared params/alice_m2/wio_ekf.yaml): /odometry/filtered
and odom -> base_footprint, the same as on robots without lidars.

lidar_correction:=true keeps the shared files untouched and only renames at launch:
  EKF            frame odom_ekf, topic /odometry/ekf, TF odom_ekf -> base_footprint
  Kinematic-ICP  TF odom -> odom_ekf (lidar correction), and /odometry/filtered in
                 frame odom = EKF odometry with the correction applied
so SLAM, Nav2, docking and the health monitor keep using /odometry/filtered and odom.
Kinematic-ICP parameters: config/alice_m2.yaml (kicp_config:= to override).

ekf_params and kicp_config take one file or a comma-separated list applied in order
(later files override earlier ones), e.g. the Isaac sim passes the shared EKF file plus
its own overlay and config/alice_m2.yaml plus its scan topics:
  ros2 launch kinematic_icp alice_m2_odometry.launch.py use_sim_time:=true \
      kicp_config:=<share>/config/alice_m2.yaml,<sim>/kicp_sim.yaml
"""
import os
import sys
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from m2_lidar_driver_params import lidar_blind_sectors  # noqa: E402


def _node_params_file(node, params):
    """params for one node as a temporary YAML file keyed by the node name.

    A launch dict goes in as a "/**" entry. rclcpp applies the entries in the order
    each key first appears, so the dict wins over wio_ekf.yaml's "ekf_filter_node:"
    only while no "/**" entry comes before that file. A global parameter set by the
    caller (SetParameter, e.g. use_sim_time in the Isaac sim's m2_sim.launch.py)
    creates "/**" first, and then the yaml's odom wins: the EKF kept publishing
    odom -> base_footprint (sim, 2026-10-13: "/odometry/ekf is in frame odom, expected
    odom_ekf"; the robot bags of 10-01/10-02 have odom_ekf, no global parameter there).
    Under the same key the later file wins whatever the caller sets, so this goes last.
    """
    with tempfile.NamedTemporaryFile("w", prefix=f"alice_m2_odometry_{node}_", suffix=".yaml",
                                     delete=False) as f:
        yaml.safe_dump({node: {"ros__parameters": params}}, f)
        return f.name


def _files(value):
    """Comma-separated parameter files, applied in order."""
    return [p.strip() for p in value.split(",") if p.strip()]


def _nodes(context):
    rl_share = get_package_share_directory("robot_localization")
    kicp_share = get_package_share_directory("kinematic_icp")
    # robot_localization keeps one wio_ekf.yaml per generation (params/alice_m2/); older
    # installs had a single params/wio_ekf.yaml.
    ekf_candidates = [os.path.join(rl_share, "params", "alice_m2", "wio_ekf.yaml"),
                      os.path.join(rl_share, "params", "wio_ekf.yaml")]
    ekf_params = _files(LaunchConfiguration("ekf_params").perform(context)) or [next(
        (p for p in ekf_candidates if os.path.isfile(p)), ekf_candidates[0])]
    kicp_config = _files(LaunchConfiguration("kicp_config").perform(context)) or [
        os.path.join(kicp_share, "config", "alice_m2.yaml")]
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context) == "true"
    correction = LaunchConfiguration("lidar_correction").perform(context) == "true"

    if not correction:
        return [Node(
            package="robot_localization", executable="ekf_node", name="ekf_filter_node",
            output="screen", parameters=[*ekf_params, {"use_sim_time": use_sim_time}],
        )]

    ekf_frame, frame = "odom_ekf", "odom"
    ekf_topic, topic = "/odometry/ekf", "/odometry/filtered"
    kicp_overrides = {
        "use_sim_time": use_sim_time,
        "wheel_odom_frame": ekf_frame,
        "lidar_odom_frame": frame,
        "publish_odom_tf": False,
        "publish_correction_tf": True,
        "corrected_odometry_input_topic": ekf_topic,
        "corrected_odometry_output_topic": topic,
    }
    # Blind sectors from the lidar driver's parameter files (see m2_lidar_driver_params).
    blind = lidar_blind_sectors()
    if blind:
        kicp_overrides["lidar_blind_sectors"] = blind
    return [
        Node(
            package="robot_localization", executable="ekf_node", name="ekf_filter_node",
            output="screen",
            parameters=[*ekf_params, _node_params_file("ekf_filter_node", {
                "use_sim_time": use_sim_time, "odom_frame": ekf_frame, "world_frame": ekf_frame})],
            remappings=[("odometry/filtered", ekf_topic)],
        ),
        Node(
            package="kinematic_icp", executable="kinematic_icp_online_node", name="online_node",
            namespace="kinematic_icp", output="screen",
            parameters=[*kicp_config, kicp_overrides],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("lidar_correction", default_value="true", choices=["true", "false"]),
        DeclareLaunchArgument("ekf_params", default_value="",
                              description="EKF parameter file(s), comma-separated (default:"
                                          " robot_localization params/alice_m2/wio_ekf.yaml)"),
        DeclareLaunchArgument("kicp_config", default_value="",
                              description="Kinematic-ICP parameter file(s), comma-separated"
                                          " (default: config/alice_m2.yaml)"),
        DeclareLaunchArgument("use_sim_time", default_value="false", choices=["true", "false"]),
        OpaqueFunction(function=_nodes),
    ])
