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
#include "Registration.hpp"

#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/global_control.h>
#include <tbb/info.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <kiss_icp/core/VoxelHashMap.hpp>
#include <limits>
#include <numeric>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <tuple>

using LinearSystem = std::pair<Eigen::Matrix2d, Eigen::Vector2d>;
using Correspondences = tbb::concurrent_vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>;

namespace {
constexpr double epsilon = std::numeric_limits<double>::min();

double ComputeOdometryRegularization(const Correspondences &associations,
                                     const Sophus::SE3d &odometry_initial_guess) {
    const double sum_of_squared_residuals =
        std::transform_reduce(associations.cbegin(), associations.cend(), 0.0, std::plus<double>(),
                              [&](const auto &association) {
                                  const auto &[source, target] = association;
                                  return (odometry_initial_guess * source - target).squaredNorm();
                              });
    const double N = static_cast<double>(associations.size());
    const double mean_squared_residual = sum_of_squared_residuals / N;
    const double beta = 1.0 / (mean_squared_residual + epsilon);
    return beta;
}

Correspondences DataAssociation(const std::vector<Eigen::Vector3d> &points,
                                const kiss_icp::VoxelHashMap &voxel_map,
                                const Sophus::SE3d &T,
                                const double max_correspondance_distance) {
    using points_iterator = std::vector<Eigen::Vector3d>::const_iterator;
    Correspondences correspondences;
    correspondences.reserve(points.size());
    tbb::parallel_for(
        // Range
        tbb::blocked_range<points_iterator>{points.cbegin(), points.cend()},
        [&](const tbb::blocked_range<points_iterator> &r) {
            std::for_each(r.begin(), r.end(), [&](const auto &point) {
                const auto &[closest_neighbor, distance] = voxel_map.GetClosestNeighbor(T * point);
                if (distance < max_correspondance_distance) {
                    correspondences.emplace_back(point, closest_neighbor);
                }
            });
        });
    return correspondences;
}

Eigen::Vector2d ComputePerturbation(const Correspondences &correspondences,
                                    const Sophus::SE3d &current_estimate,
                                    const double beta) {
    auto compute_jacobian_and_residual = [&](const auto &correspondence) {
        const auto &[source, target] = correspondence;
        const Eigen::Vector3d residual = current_estimate * source - target;
        Eigen::Matrix<double, 3, 2> J;
        J.col(0) = current_estimate.so3() * Eigen::Vector3d::UnitX();
        J.col(1) = current_estimate.so3() * Eigen::Vector3d(-source.y(), source.x(), 0.0);
        return std::make_tuple(J, residual);
    };

    auto sum_linear_systems = [](LinearSystem a, const LinearSystem &b) {
        a.first += b.first;
        a.second += b.second;
        return a;
    };

    using correspondence_iterator = Correspondences::const_iterator;
    auto [JTJ, JTr] = tbb::parallel_reduce(
        // Range
        tbb::blocked_range<correspondence_iterator>{correspondences.cbegin(),
                                                    correspondences.cend()},
        // Identity
        LinearSystem(Eigen::Matrix2d::Zero(), Eigen::Vector2d::Zero()),
        // 1st Lambda: Parallel computation
        [&](const tbb::blocked_range<correspondence_iterator> &r, LinearSystem J) -> LinearSystem {
            return std::transform_reduce(
                r.begin(), r.end(), J, sum_linear_systems, [&](const auto &correspondence) {
                    const auto &[J_r, residual] = compute_jacobian_and_residual(correspondence);
                    return LinearSystem(J_r.transpose() * J_r,        // JTJ
                                        J_r.transpose() * residual);  // JTr
                });
        },
        // 2nd Lambda: Parallel reduction of the private Jacboians
        sum_linear_systems);
    const double num_correspondences = static_cast<double>(correspondences.size());

    const Eigen::Matrix2d Omega = Eigen::Vector2d(beta, 0).asDiagonal();
    JTJ /= num_correspondences;
    JTr /= num_correspondences;
    JTJ += Omega;
    return -(JTJ.inverse() * JTr);
}

// Holonomic (x, y, yaw) counterpart of the normal equations above, normalized by
// the number of correspondences the same way. The perturbation is applied on the
// right, T * exp(dx, dy, 0, 0, 0, dyaw), so the Jacobian columns are the body x
// and y axes and the yaw lever arm of each source point, all rotated into the
// map frame - the same convention as the two-parameter version. A match with a
// normal (robot frame) contributes the scalar residual n . (T s - t) instead.
struct Match {
    Eigen::Vector3d source, target, normal;  // normal zero: point-to-point
};
using Matches = tbb::concurrent_vector<Match>;
using LinearSystem3 = std::pair<Eigen::Matrix3d, Eigen::Vector3d>;

Matches AssociateWithNormals(const std::vector<Eigen::Vector3d> &points,
                             const std::vector<Eigen::Vector3d> &normals,
                             const kiss_icp::VoxelHashMap &voxel_map,
                             const Sophus::SE3d &T,
                             const double max_correspondance_distance) {
    Matches matches;
    matches.reserve(points.size());
    tbb::parallel_for(tbb::blocked_range<std::size_t>{0, points.size()},
                      [&](const tbb::blocked_range<std::size_t> &r) {
                          for (std::size_t i = r.begin(); i < r.end(); ++i) {
                              const auto &[closest_neighbor, distance] =
                                  voxel_map.GetClosestNeighbor(T * points[i]);
                              if (distance < max_correspondance_distance) {
                                  matches.push_back({points[i], closest_neighbor,
                                                     normals.empty() ? Eigen::Vector3d::Zero()
                                                                     : normals[i]});
                              }
                          }
                      });
    return matches;
}

struct MatchSystem {
    LinearSystem3 system{Eigen::Matrix3d::Zero(), Eigen::Vector3d::Zero()};
    double sum_squared_residual = 0.0;
    std::size_t used = 0, point_to_line = 0;
};

MatchSystem BuildLinearSystem3(const Matches &matches,
                               const Sophus::SE3d &T,
                               const double point_to_point_weight) {
    const Eigen::Matrix3d R = T.so3().matrix();
    auto reduce = [](MatchSystem a, const MatchSystem &b) {
        a.system.first += b.system.first;
        a.system.second += b.system.second;
        a.sum_squared_residual += b.sum_squared_residual;
        a.used += b.used;
        a.point_to_line += b.point_to_line;
        return a;
    };
    MatchSystem out = tbb::parallel_reduce(
        tbb::blocked_range<Matches::const_iterator>{matches.cbegin(), matches.cend()},
        MatchSystem{},
        [&](const tbb::blocked_range<Matches::const_iterator> &r, MatchSystem acc) -> MatchSystem {
            for (auto it = r.begin(); it != r.end(); ++it) {
                const Eigen::Vector3d &s = it->source;
                Eigen::Matrix3d J;
                J.col(0) = R * Eigen::Vector3d::UnitX();
                J.col(1) = R * Eigen::Vector3d::UnitY();
                J.col(2) = R * Eigen::Vector3d(-s.y(), s.x(), 0.0);
                const Eigen::Vector3d residual = T * s - it->target;
                if (it->normal.squaredNorm() > 0.5) {
                    const Eigen::Vector3d n = R * it->normal;
                    const Eigen::RowVector3d Jn = n.transpose() * J;
                    const double rn = n.dot(residual);
                    acc.system.first += Jn.transpose() * Jn;
                    acc.system.second += Jn.transpose() * rn;
                    acc.sum_squared_residual += rn * rn;
                    ++acc.used;
                    ++acc.point_to_line;
                } else if (point_to_point_weight > 0.0) {
                    acc.system.first += point_to_point_weight * (J.transpose() * J);
                    acc.system.second += point_to_point_weight * (J.transpose() * residual);
                    acc.sum_squared_residual += residual.squaredNorm();
                    ++acc.used;
                }
            }
            return acc;
        },
        reduce);
    if (out.used > 0) {
        out.system.first /= static_cast<double>(out.used);
        out.system.second /= static_cast<double>(out.used);
    }
    return out;
}

}  // namespace

namespace kinematic_icp {

KinematicRegistration::KinematicRegistration(const int max_num_iteration,
                                             const double convergence_criterion,
                                             const int max_num_threads,
                                             const bool use_adaptive_odometry_regularization,
                                             const double fixed_regularization,
                                             const HolonomicOptions &holonomic)
    : max_num_iterations_(max_num_iteration),
      convergence_criterion_(convergence_criterion),
      // Only manipulate the number of threads if the user specifies something
      // greater than 0
      max_num_threads_(max_num_threads > 0 ? max_num_threads
                                           : tbb::this_task_arena::max_concurrency()),
      use_adaptive_odometry_regularization_(use_adaptive_odometry_regularization),
      fixed_regularization_(fixed_regularization),
      holonomic_(holonomic) {
    holonomic_.damping_weights = holonomic_.damping_weights.cwiseMax(0.0);
    // This global variable requires static duration storage to be able to
    // manipulate the max concurrency from TBB across the entire class
    static const auto tbb_control_settings = tbb::global_control(
        tbb::global_control::max_allowed_parallelism, static_cast<size_t>(max_num_threads_));
}

Sophus::SE3d KinematicRegistration::ComputeRobotMotion(const std::vector<Eigen::Vector3d> &frame,
                                                       const std::vector<Eigen::Vector3d> &normals,
                                                       const kiss_icp::VoxelHashMap &voxel_map,
                                                       const Sophus::SE3d &last_robot_pose,
                                                       const Sophus::SE3d &relative_wheel_odometry,
                                                       const double max_correspondence_distance) {
    Sophus::SE3d current_estimate = last_robot_pose * relative_wheel_odometry;
    diagnostics_ = RegistrationDiagnostics{};
    if (voxel_map.Empty()) return current_estimate;

    if (holonomic_.enabled) {
        const Sophus::SE3d prior_pose = current_estimate;
        const std::vector<Eigen::Vector3d> no_normals;
        const auto &match_normals = holonomic_.point_to_line ? normals : no_normals;
        // Point-to-point matches always count; with point_to_line, points without
        // a normal count only through point_to_point_weight.
        const double p2p_weight =
            (holonomic_.point_to_line && !match_normals.empty()) ? holonomic_.point_to_point_weight
                                                                  : 1.0;
        auto matches = AssociateWithNormals(frame, match_normals, voxel_map, current_estimate,
                                            max_correspondence_distance);
        if (matches.empty()) return current_estimate;

        // Odometry term, see HolonomicOptions.
        Eigen::Vector3d inv_prior_var = Eigen::Vector3d::Zero();
        double beta = 0.0;
        if (holonomic_.odometry_prior) {
            const Sophus::SE3d::Tangent d = relative_wheel_odometry.log();
            const Eigen::Vector3d sigma(
                holonomic_.prior_sigma_xy_floor + holonomic_.prior_sigma_x_rel * std::abs(d(0)),
                holonomic_.prior_sigma_xy_floor + holonomic_.prior_sigma_y_rel * std::abs(d(1)),
                holonomic_.prior_sigma_yaw_floor + holonomic_.prior_sigma_yaw_rel * std::abs(d(5)));
            inv_prior_var = sigma.cwiseMax(1e-9).cwiseAbs2().cwiseInverse();
        } else if (use_adaptive_odometry_regularization_) {
            // Upstream's beta: inverse mean squared point residual at the prior.
            double sum = 0.0;
            for (const auto &m : matches) sum += (current_estimate * m.source - m.target).squaredNorm();
            beta = 1.0 / (sum / static_cast<double>(matches.size()) + epsilon);
        } else {
            beta = fixed_regularization_;
        }
        auto omega_diagonal = [&](const std::size_t n) -> Eigen::Vector3d {
            if (!holonomic_.odometry_prior) return beta * holonomic_.damping_weights;
            const double point_var = holonomic_.point_sigma * holonomic_.point_sigma;
            return (point_var / static_cast<double>(std::max<std::size_t>(n, 1))) * inv_prior_var;
        };
        auto deviation_from_prior = [&](const Sophus::SE3d &T) -> Eigen::Vector3d {
            const Sophus::SE3d::Tangent e = (prior_pose.inverse() * T).log();
            return Eigen::Vector3d(e(0), e(1), e(5));
        };

        Eigen::Matrix3d Omega = Eigen::Matrix3d::Zero();
        for (int j = 0; j < max_num_iterations_; ++j) {
            const MatchSystem ms = BuildLinearSystem3(matches, current_estimate, p2p_weight);
            if (ms.used == 0) break;
            const auto &[JTJ, JTr] = ms.system;
            Omega = omega_diagonal(ms.used).asDiagonal();
            Eigen::Vector3d rhs = JTr;
            if (holonomic_.odometry_prior) rhs += Omega * deviation_from_prior(current_estimate);
            const Eigen::Vector3d dx = -(JTJ + Omega).ldlt().solve(rhs);
            if (!dx.allFinite()) break;
            Sophus::SE3d::Tangent xi = Sophus::SE3d::Tangent::Zero();
            xi(0) = dx(0);
            xi(1) = dx(1);
            xi(5) = dx(2);
            current_estimate = current_estimate * Sophus::SE3d::exp(xi);
            if (dx.norm() < convergence_criterion_) break;
            matches = AssociateWithNormals(frame, match_normals, voxel_map, current_estimate,
                                           max_correspondence_distance);
            if (matches.empty()) break;
        }
        const MatchSystem ms = BuildLinearSystem3(matches, current_estimate, p2p_weight);
        Omega = omega_diagonal(ms.used).asDiagonal();
        diagnostics_.information = ms.system.first + Omega;
        diagnostics_.mean_squared_residual =
            ms.used > 0 ? ms.sum_squared_residual / static_cast<double>(ms.used) : 0.0;
        diagnostics_.num_correspondences = ms.used;
        diagnostics_.num_point_to_line = ms.point_to_line;
        diagnostics_.omega_diagonal = Omega.diagonal();
        return current_estimate;
    }

    auto motion_model = [](const Eigen::Vector2d &integrated_controls) {
        Sophus::SE3d::Tangent dx = Sophus::SE3d::Tangent::Zero();
        const double &displacement = integrated_controls(0);
        const double &theta = integrated_controls(1);
        dx(0) = displacement * std::sin(theta) / (theta + epsilon);
        dx(1) = displacement * (1.0 - std::cos(theta)) / (theta + epsilon);
        dx(5) = theta;
        return Sophus::SE3d::exp(dx);
    };
    auto correspondences =
        DataAssociation(frame, voxel_map, current_estimate, max_correspondence_distance);

    const double regularization_term = [&]() {
        if (use_adaptive_odometry_regularization_) {
            return ComputeOdometryRegularization(correspondences, current_estimate);
        } else {
            return fixed_regularization_;
        }
    }();
    // ICP-loop
    for (int j = 0; j < max_num_iterations_; ++j) {
        const auto dx = ComputePerturbation(correspondences, current_estimate, regularization_term);
        const auto delta_motion = motion_model(dx);
        current_estimate = current_estimate * delta_motion;
        // Break loop
        if (dx.norm() < convergence_criterion_) break;
        correspondences =
            DataAssociation(frame, voxel_map, current_estimate, max_correspondence_distance);
    }
    // Spit the final transformation
    return current_estimate;
}

std::vector<Eigen::Vector3d> KinematicRegistration::EstimateNormals(
    const std::vector<Eigen::Vector3d> &points, const std::vector<Eigen::Vector3d> &cloud) const {
    std::vector<Eigen::Vector3d> normals(points.size(), Eigen::Vector3d::Zero());
    const double r2 = holonomic_.normal_radius * holonomic_.normal_radius;
    tbb::parallel_for(tbb::blocked_range<std::size_t>{0, points.size()},
                      [&](const tbb::blocked_range<std::size_t> &r) {
                          for (std::size_t i = r.begin(); i < r.end(); ++i) {
                              const Eigen::Vector2d p = points[i].head<2>();
                              Eigen::Vector2d sum = Eigen::Vector2d::Zero();
                              Eigen::Matrix2d sum_sq = Eigen::Matrix2d::Zero();
                              int count = 0;
                              for (const auto &c : cloud) {
                                  const Eigen::Vector2d q = c.head<2>();
                                  if ((q - p).squaredNorm() > r2) continue;
                                  sum += q;
                                  sum_sq += q * q.transpose();
                                  ++count;
                              }
                              if (count < holonomic_.normal_min_points) continue;
                              const Eigen::Vector2d mean = sum / count;
                              const Eigen::Matrix2d cov = sum_sq / count - mean * mean.transpose();
                              const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(cov);
                              const double l_min = es.eigenvalues()(0), l_max = es.eigenvalues()(1);
                              if (!(l_max > 1e-12) || l_min / l_max > holonomic_.normal_max_eigen_ratio) {
                                  continue;
                              }
                              const Eigen::Vector2d n = es.eigenvectors().col(0).normalized();
                              normals[i] = Eigen::Vector3d(n.x(), n.y(), 0.0);
                          }
                      });
    return normals;
}

}  // namespace kinematic_icp
