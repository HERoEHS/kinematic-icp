#!/usr/bin/env python3
"""ALICE M2: synthetic rosbag for an end-to-end check of the Kinematic-ICP ROS layer.

Writes /ydlidar_front/scan, /ydlidar_rear/scan (LaserScan, 10 Hz each, with a
phase offset and different latencies so the two streams interleave and can arrive
out of order), /tf (odom -> base_footprint at 50 Hz from a wheel-like prior:
+0.8 % along x, -4 % along y while crabbing, gyro-accurate yaw) and /tf_static
(base_footprint -> base_link and the two lidar mounts, flipped as on the robot).
Ground truth goes to <out>_gt_tum.txt and the prior to <out>_prior_tum.txt.

  make_synthetic_bag.py OUT_DIR [hall|corridor] [ydlidar|index] [seed] [ekf_frame]

ekf_frame (default odom): frame of the EKF-like prior. With odom_ekf the bag
mimics alice_m2_odometry.launch.py lidar_correction:=true: TF odom_ekf ->
base_footprint and the prior as nav_msgs/Odometry on /odometry/ekf (50 Hz);
otherwise TF odom -> base_footprint and /odometry/filtered.

Beam timing: "ydlidar" (default) emulates the YDLidar ROS 2 driver as read from
its source: one revolution is measured from scan angle 0 deg with the angle
increasing, header.stamp is the first measured node, and the beams are binned by
angle into [-180, 180) deg - so array order is half a revolution off measurement
order. "index" makes array order the measurement order, which is what
laser_geometry assumes.
"""
import math
import sys

import numpy as np
import rclpy.serialization as ser
import rosbag2_py
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage

DT = 0.001
T0 = 1_000_000_000.0  # s, epoch offset of the synthetic clock

# Mounts as in ydlidar_dual.launch.py (2026-10-01), parent base_link.
MOUNTS = {
    "front": dict(frame="laser_front_frame", topic="/ydlidar_front/scan",
                  xyz=(0.4158, -0.297, 0.0813), rpy=(0.0, math.pi, 0.349066),
                  phase=0.000, latency=0.030),
    "rear": dict(frame="laser_rear_frame", topic="/ydlidar_rear/scan",
                 xyz=(-0.4858, 0.297, 0.0813), rpy=(0.0, math.pi, 3.49066),
                 phase=0.037, latency=0.075),
}
BASE_LINK_Z = 0.072
N_BEAMS = 430                      # 360 deg at 0.839 deg, as the driver publishes
IGNORE_DEG = (17.0, 110.0)         # ignore_array, scan angle frame
SCAN_PERIOD = 0.1
RMAX, SIGMA_R = 12.0, 0.015


def rot_rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def quat_from_R(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return x, y, z, w


def box(w, x0, y0, x1, y1):
    w += [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]


def world(name):
    w = []
    if name == "hall":
        box(w, 0, 0, 16, 12)
        for b in [(8.8, 3.8, 9.2, 4.2), (10.8, 8.8, 11.2, 9.2), (12.8, 4.8, 13.2, 5.2), (2.8, 9.8, 3.2, 10.2),
                  (14, 1, 15, 2)]:
            box(w, *b)
        w.append(((7, 11), (11, 11)))
        phases = [(5.0, 0.5, 0, 0), (5.0, 0, 0.5, 0), (math.pi, 0, 0, 0.5), (6.0, 0.35, 0.35, 0),
                  (6.0, 0, -0.5, 0), (6.0, 0.4, 0, -0.3)]
        start = (2.5, 2.5, 0.0)
    else:
        width, depth, door = 1.0, 0.15, 0.9
        for side in (0, 1):
            y = 0.0 if side == 0 else width
            out = -depth if side == 0 else depth
            x = -2.0
            d = 5.0 + 3.5 * side
            while d < 40.0:
                w += [((x, y), (d, y)), ((d, y), (d, y + out)), ((d, y + out), (d + door, y + out)),
                      ((d + door, y + out), (d + door, y))]
                x = d + door
                d += 7.0
            w.append(((x, y), (40.0, y)))
        w += [((-2, 0), (-2, width)), ((40, 0), (40, width))]
        phases = [(24.0, 0.5, 0, 0), (10.0, -0.5, 0, 0)]
        start = (1.0, 0.5, 0.0)
    A = np.array([s[0] for s in w], float)
    B = np.array([s[1] for s in w], float)
    return A, B, phases, start


def integrate(phases, start):
    P = [np.array(start, float)]
    crab = [False]
    v = np.zeros(3)
    for dur, vx, vy, wz in phases:
        v0 = v.copy()
        tgt = np.array([vx, vy, wz])
        for i in range(int(round(dur / DT))):
            v = v0 + min(1.0, i * DT / 0.3) * (tgt - v0)
            x, y, th = P[-1]
            P.append(np.array([x + (v[0] * math.cos(th) - v[1] * math.sin(th)) * DT,
                               y + (v[0] * math.sin(th) + v[1] * math.cos(th)) * DT,
                               math.atan2(math.sin(th + v[2] * DT), math.cos(th + v[2] * DT))]))
            crab.append(abs(vy) > 0.05)
    return np.array(P), np.array(crab)


def raycast(A, B, o, d):
    e = B - A
    den = d[0] * e[:, 1] - d[1] * e[:, 0]
    q = A - o
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (q[:, 0] * e[:, 1] - q[:, 1] * e[:, 0]) / den
        u = (q[:, 0] * d[1] - q[:, 1] * d[0]) / den
    ok = (np.abs(den) > 1e-12) & (t > 1e-6) & (u >= 0) & (u <= 1)
    if not ok.any():
        return -1.0
    tm = t[ok].min()
    return tm if tm <= RMAX else -1.0


def stamp(t):
    sec = int(math.floor(t))
    return Time(sec=sec, nanosec=int(round((t - sec) * 1e9)) % 1_000_000_000)


def tf_msg(t, parent, child, xyz, q):
    m = TransformStamped()
    m.header.stamp = stamp(t)
    m.header.frame_id = parent
    m.child_frame_id = child
    m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = map(float, xyz)
    m.transform.rotation.x, m.transform.rotation.y, m.transform.rotation.z, m.transform.rotation.w = map(float, q)
    return m


def main():
    out = sys.argv[1]
    scen = sys.argv[2] if len(sys.argv) > 2 else "hall"
    timing = sys.argv[3] if len(sys.argv) > 3 else "ydlidar"
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 7
    ekf_frame = sys.argv[5] if len(sys.argv) > 5 else "odom"
    odom_topic = "/odometry/ekf" if ekf_frame != "odom" else "/odometry/filtered"
    A, B, phases, start = world(scen)
    P, crab = integrate(phases, start)
    n = len(P)
    rng = np.random.default_rng(seed)

    # Prior (EKF-like): integrate distorted body increments at 1 ms.
    prior = np.zeros_like(P)
    prior[0] = P[0]
    for k in range(1, n):
        x0, y0, t0 = P[k - 1]
        dxw, dyw = P[k, 0] - x0, P[k, 1] - y0
        dx = math.cos(t0) * dxw + math.sin(t0) * dyw
        dy = -math.sin(t0) * dxw + math.cos(t0) * dyw
        dth = math.atan2(math.sin(P[k, 2] - t0), math.cos(P[k, 2] - t0))
        dx *= 1.008
        dy *= 0.96 if crab[k] else 1.008
        dth *= 1.003
        px, py, pt = prior[k - 1]
        prior[k] = (px + dx * math.cos(pt) - dy * math.sin(pt), py + dx * math.sin(pt) + dy * math.cos(pt),
                    math.atan2(math.sin(pt + dth), math.cos(pt + dth)))

    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=out + "/bag", storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    for i, (topic, typ) in enumerate([("/tf", "tf2_msgs/msg/TFMessage"), ("/tf_static", "tf2_msgs/msg/TFMessage"),
                                      (odom_topic, "nav_msgs/msg/Odometry"),
                                      (MOUNTS["front"]["topic"], "sensor_msgs/msg/LaserScan"),
                                      (MOUNTS["rear"]["topic"], "sensor_msgs/msg/LaserScan")]):
        writer.create_topic(rosbag2_py.TopicMetadata(id=i, name=topic, type=typ, serialization_format="cdr"))
    msgs = []  # (receive time, topic, msg)

    statics = TFMessage()
    statics.transforms.append(tf_msg(T0, "base_footprint", "base_link", (0, 0, BASE_LINK_Z), (0, 0, 0, 1)))
    for m in MOUNTS.values():
        m["R"] = rot_rpy(*m["rpy"])
        statics.transforms.append(tf_msg(T0, "base_link", m["frame"], m["xyz"], quat_from_R(m["R"])))
    msgs.append((T0, "/tf_static", statics))

    for k in range(0, n, 20):  # 50 Hz
        t = T0 + k * DT
        x, y, th = prior[k]
        msgs.append((t + 0.002, "/tf", TFMessage(transforms=[
            tf_msg(t, ekf_frame, "base_footprint", (x, y, 0), (0, 0, math.sin(th / 2), math.cos(th / 2)))])))
        od = Odometry()
        od.header.stamp = stamp(t)
        od.header.frame_id = ekf_frame
        od.child_frame_id = "base_footprint"
        od.pose.pose.position.x, od.pose.pose.position.y = float(x), float(y)
        od.pose.pose.orientation.z, od.pose.pose.orientation.w = math.sin(th / 2), math.cos(th / 2)
        od.pose.covariance[0] = od.pose.covariance[7] = 0.01
        od.pose.covariance[35] = 0.001
        msgs.append((t + 0.002, odom_topic, od))

    ang = np.radians(-180.0 + 360.0 * np.arange(N_BEAMS) / N_BEAMS)
    ign = (np.degrees(ang) >= IGNORE_DEG[0]) & (np.degrees(ang) <= IGNORE_DEG[1])
    tinc = SCAN_PERIOD / N_BEAMS
    for name, m in MOUNTS.items():
        Rl = m["R"]
        tl = np.array(m["xyz"]) + np.array([0, 0, BASE_LINK_Z])
        t_scan = m["phase"]
        while t_scan + SCAN_PERIOD < (n - 1) * DT:
            ranges = np.zeros(N_BEAMS)
            for i in range(N_BEAMS):
                if timing == "ydlidar":
                    frac = (math.degrees(ang[i]) % 360.0) / 360.0   # measured from 0 deg, increasing
                else:
                    frac = i / N_BEAMS
                k = int(round((t_scan + frac * SCAN_PERIOD) / DT))
                x, y, th = P[k]
                c, s = math.cos(th), math.sin(th)
                Rb = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                o = Rb @ tl + np.array([x, y, 0])
                d = Rb @ Rl @ np.array([math.cos(ang[i]), math.sin(ang[i]), 0.0])
                if ign[i]:
                    continue
                r = raycast(A, B, o[:2], d[:2] / np.linalg.norm(d[:2]))
                if r > 0:
                    ranges[i] = r + rng.normal(0, SIGMA_R)
            msg = LaserScan()
            msg.header.stamp = stamp(T0 + t_scan)           # first point, like the driver
            msg.header.frame_id = m["frame"]
            msg.angle_min, msg.angle_max = float(ang[0]), float(ang[-1])
            msg.angle_increment = float(ang[1] - ang[0])
            msg.time_increment = float(tinc)
            msg.scan_time = SCAN_PERIOD
            msg.range_min, msg.range_max = 0.03, RMAX
            msg.ranges = ranges.astype(np.float32).tolist()
            msgs.append((T0 + t_scan + SCAN_PERIOD + m["latency"], m["topic"], msg))
            t_scan += SCAN_PERIOD

    msgs.sort(key=lambda r: r[0])
    for t, topic, msg in msgs:
        writer.write(topic, ser.serialize_message(msg), int(t * 1e9))
    del writer

    def tum(path, Q):
        with open(path, "w") as f:
            for k in range(0, n, 10):
                x, y, th = Q[k]
                f.write(f"{T0 + k * DT:.6f} {x:.6f} {y:.6f} 0 0 0 {math.sin(th / 2):.9f} {math.cos(th / 2):.9f}\n")
    tum(out + "/gt_tum.txt", P)
    tum(out + "/prior_tum.txt", prior)
    print(f"{scen} ({timing} timing): {n * DT:.1f} s, {sum(1 for r in msgs if 'scan' in r[1])} scans -> {out}/bag")


if __name__ == "__main__":
    main()
