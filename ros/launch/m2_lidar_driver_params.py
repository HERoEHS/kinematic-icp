"""ALICE M2: lidar settings that live with the lidar driver, for the Kinematic-ICP launch files.

lidar_blind_sectors() reads every parameter file of ydlidar_ros2_driver (the files
ydlidar_dual.launch.py starts the drivers with) and returns "frame_id:ignore_array"
for each frame, the value of the Kinematic-ICP parameter lidar_blind_sectors. The
online node then also asks the running driver, so a driver started with another
file still gets its own sectors.
"""
import glob
import os

import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory


def lidar_blind_sectors(package="ydlidar_ros2_driver"):
    try:
        share = get_package_share_directory(package)
    except PackageNotFoundError:
        return []
    found = {}
    for path in sorted(glob.glob(os.path.join(share, "params", "*.yaml"))):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        for node in data.values() if isinstance(data, dict) else []:
            params = node.get("ros__parameters", {}) if isinstance(node, dict) else {}
            if "frame_id" not in params or "ignore_array" not in params:
                continue
            frame, sectors = str(params["frame_id"]), str(params["ignore_array"])
            if found.get(frame, sectors) != sectors:
                print(f"[m2_lidar_driver_params] {frame}: ignore_array differs between driver "
                      f"parameter files, using {path}")
            found[frame] = sectors
    return [f"{frame}:{sectors}" for frame, sectors in found.items()]
