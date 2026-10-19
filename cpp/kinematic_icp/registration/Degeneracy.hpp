// ALICE M2 (2026-10-19): how well the lidar alone pins the robot position.
//
// In a corridor every line normal points across it, so the scans carry no
// information about the position along it; in a wide open space few walls are in
// range at all. The holonomic registration then keeps the odometry prior in that
// direction (HolonomicOptions), which is the right thing to do, but nothing
// downstream could tell that the position along that axis now rests on the wheels
// alone. This sums the scan-only information of the frames registered in the last
// `window` seconds (RegistrationDiagnostics without the odometry term, rotated into
// the odometry frame so the frames of a turning robot add up) and judges it:
//
//  - corridor: lambda_min / lambda_max of the 2D position information below
//    ratio_enter (released above ratio_exit). Weak axis = eigenvector of lambda_min.
//  - sparse: fewer than min_matches_enter point-to-line matches per registered frame
//    (released at min_matches_exit), e.g. a hall whose walls are out of range.
//
// The sigmas treat every match as independent with sigma point_sigma, so they are
// optimistic; use them to compare directions, not as a calibrated uncertainty.
#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <deque>
#include <string>

namespace kinematic_icp {

struct DegeneracyOptions {
    double window = 1.0;           // [s] registered frames summed (both lidars: ~20)
    double ratio_enter = 0.05;     // lambda_min / lambda_max of the position information
    double ratio_exit = 0.10;
    double min_matches_enter = 8.0;  // mean matches per registered frame
    double min_matches_exit = 12.0;
    // Fewer registered frames in the window are not judged. Right after start-up the first
    // scans meet an almost empty local map (few matches); judging those alone called the
    // start pose sparse, and a robot that then stood still kept that label (Isaac 2026-10-20).
    std::size_t min_frames = 5;
};

enum class DegeneracyReason { kNone, kCorridor, kSparse };

struct DegeneracyState {
    bool degenerate = false;
    DegeneracyReason reason = DegeneracyReason::kNone;
    double ratio = 0.0;                                // lambda_min / lambda_max, 0..1
    Eigen::Vector2d weak_axis = Eigen::Vector2d::UnitX();  // unit, odometry frame
    double sigma_weak = 0.0;                           // [m] along weak_axis (0 = no information)
    double sigma_strong = 0.0;                         // [m] across it
    double sigma_yaw = 0.0;                            // [rad] with the position known
    double matches_per_frame = 0.0;
    std::size_t frames = 0;                            // registered frames in the window
    bool judged = false;                               // frames >= min_frames
};

const char *ToString(DegeneracyReason reason);

class LidarDegeneracy {
public:
    explicit LidarDegeneracy(const DegeneracyOptions &options = DegeneracyOptions{});

    // One registered frame. information_per_match: the scan-only normalized information
    // in (x, y, yaw) of the robot frame, i.e. RegistrationDiagnostics::information minus
    // the odometry term (zero when nothing matched); point_sigma scales it to absolute
    // information. yaw: robot heading in the odometry frame [rad].
    void AddFrame(double stamp,
                  double yaw,
                  const Eigen::Matrix3d &information_per_match,
                  std::size_t matches,
                  double point_sigma);

    bool Empty() const { return frames_.empty(); }
    const DegeneracyState &state() const { return state_; }
    // Weak axis as an undirected angle in [0, 180) degrees in the odometry frame.
    double WeakAxisDegrees() const;

private:
    struct Frame {
        double stamp;
        Eigen::Matrix3d information;  // absolute, odometry frame axes
        std::size_t matches;
    };
    void Update();

    DegeneracyOptions options_;
    std::deque<Frame> frames_;
    DegeneracyState state_;
};

}  // namespace kinematic_icp
