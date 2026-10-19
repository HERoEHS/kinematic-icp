// ALICE M2: see Degeneracy.hpp.
#include "Degeneracy.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>

namespace kinematic_icp {

const char *ToString(const DegeneracyReason reason) {
    switch (reason) {
        case DegeneracyReason::kCorridor:
            return "corridor";
        case DegeneracyReason::kSparse:
            return "sparse";
        default:
            return "none";
    }
}

LidarDegeneracy::LidarDegeneracy(const DegeneracyOptions &options) : options_(options) {}

void LidarDegeneracy::AddFrame(const double stamp,
                               const double yaw,
                               const Eigen::Matrix3d &information_per_match,
                               const std::size_t matches,
                               const double point_sigma) {
    // Robot frame -> odometry frame. The registration perturbs the pose on the right,
    // so its x, y are robot axes: delta_odom = R delta_robot, information R H R^T.
    Eigen::Matrix3d T = Eigen::Matrix3d::Identity();
    const double c = std::cos(yaw), s = std::sin(yaw);
    T.topLeftCorner<2, 2>() << c, -s, s, c;
    Eigen::Matrix3d information = Eigen::Matrix3d::Zero();
    if (matches > 0 && point_sigma > 0.0) {
        const Eigen::Matrix3d symmetric = 0.5 * (information_per_match + information_per_match.transpose());
        information = static_cast<double>(matches) / (point_sigma * point_sigma) * T * symmetric *
                      T.transpose();
    }
    frames_.push_back({stamp, information, matches});
    // A frame stamped before the newest by more than the window (or a clock jump back)
    // ends the old window.
    while (frames_.size() > 1 && (frames_.front().stamp < stamp - options_.window ||
                                  frames_.front().stamp > stamp)) {
        frames_.pop_front();
    }
    Update();
}

void LidarDegeneracy::Update() {
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    std::size_t matches = 0;
    for (const auto &frame : frames_) {
        H += frame.information;
        matches += frame.matches;
    }
    DegeneracyState next;
    next.frames = frames_.size();
    next.matches_per_frame =
        frames_.empty() ? 0.0 : static_cast<double>(matches) / static_cast<double>(frames_.size());

    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(H.topLeftCorner<2, 2>());
    const double lambda_min = std::max(es.eigenvalues()(0), 0.0);
    const double lambda_max = std::max(es.eigenvalues()(1), 0.0);
    next.ratio = lambda_max > 0.0 ? lambda_min / lambda_max : 0.0;
    next.weak_axis = es.eigenvectors().col(0).normalized();
    next.sigma_weak = lambda_min > 0.0 ? 1.0 / std::sqrt(lambda_min) : 0.0;
    next.sigma_strong = lambda_max > 0.0 ? 1.0 / std::sqrt(lambda_max) : 0.0;
    next.sigma_yaw = H(2, 2) > 0.0 ? 1.0 / std::sqrt(H(2, 2)) : 0.0;

    next.judged = frames_.size() >= options_.min_frames;
    if (!next.judged) {  // too little to tell: not degenerate, and no hysteresis carried over
        state_ = next;
        return;
    }
    // Hysteresis: stay degenerate until both measures clear their exit thresholds.
    const bool was = state_.degenerate;
    const bool sparse = next.matches_per_frame <
                        (was ? options_.min_matches_exit : options_.min_matches_enter);
    const bool corridor = next.ratio < (was ? options_.ratio_exit : options_.ratio_enter);
    next.degenerate = sparse || corridor;
    next.reason = sparse ? DegeneracyReason::kSparse
                         : (corridor ? DegeneracyReason::kCorridor : DegeneracyReason::kNone);
    state_ = next;
}

double LidarDegeneracy::WeakAxisDegrees() const {
    double angle = std::atan2(state_.weak_axis.y(), state_.weak_axis.x()) * 180.0 / M_PI;
    if (angle < 0.0) angle += 180.0;
    if (angle >= 180.0) angle -= 180.0;
    return angle;
}

}  // namespace kinematic_icp
