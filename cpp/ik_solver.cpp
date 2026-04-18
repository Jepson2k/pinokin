#include "ik_solver.h"
#include <cmath>
#include <limits>

#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>

namespace pinokin {

static constexpr double PI = 3.14159265358979323846264338327950288;
static constexpr double PI_2 = 1.57079632679489661923132169163975144;
static constexpr double PI_x2 = 6.283185307179586;

static inline double wrapToPi(double x) { return std::atan2(std::sin(x), std::cos(x)); }

IKSolver::IKSolver(const Robot& robot, Method method, Damping damping,
                   double tol, double lambda, int max_iter, int max_restarts,
                   bool enforce_limits, double kq, double km, double ps,
                   double pi)
    : robot_(robot)
    , method_(method)
    , damping_(damping)
    , tol_(tol)
    , lambda_(lambda)
    , max_iter_(max_iter)
    , max_restarts_(max_restarts)
    , enforce_limits_(enforce_limits)
    , we_is_identity_(true)
    , kq_(kq)
    , km_(km)
    , ps_(ps)
    , pi_(pi)
    , nullspace_active_(kq > 0.0 || km > 0.0)
    , rng_(std::random_device{}())
{
    int n = robot_.nq();
    We_ = Eigen::Matrix<double, 6, 6>::Identity();
    J_.resize(6, n);
    e_.setZero();
    JtWJ_.resize(n, n);
    g_.resize(n);
    Te_ = Eigen::Matrix4d::Identity();
    q_.resize(n);
    q_.setZero();
    q0_.resize(n);
    q0_.setZero();
    success_ = false;
    residual_ = 0.0;
    iterations_ = 0;
    restarts_ = 0;

    qnull_.setZero(n);
    qnull_grad_.setZero(n);
    sigma_.setZero(n);
    Jm_.setZero(n);
    N_.setZero(n, n);
    Jpinv_.setZero(n, 6);
    J_fd_plus_.setZero(6, n);
    J_fd_minus_.setZero(6, n);
    q_perturb_.setZero(n);
}

void IKSolver::set_we(const Eigen::VectorXd& we) {
    if (we.size() != 6) {
        throw std::runtime_error("we must be a 6-vector");
    }
    We_ = we.asDiagonal();
    // Check if we is all ones (identity)
    we_is_identity_ = we.isApprox(Eigen::VectorXd::Ones(6));
}

bool IKSolver::solve(const Eigen::Matrix4d& Tep, const Eigen::VectorXd* q0) {
    success_ = false;
    iterations_ = 0;
    restarts_ = 0;
    residual_ = 0.0;

    if (q0 && q0->size() == robot_.nq()) {
        q_ = *q0;
        q0_ = *q0;
    } else {
        rand_q();
        q0_ = q_;
    }

    switch (method_) {
        case Method::GN: solve_gn(Tep); break;
        case Method::NR: solve_nr(Tep); break;
        case Method::LM: solve_lm(Tep); break;
    }

    return success_;
}

// ===== Gauss-Newton (ported from _IK_GN in ik.cpp) =====
void IKSolver::solve_gn(const Eigen::Matrix4d& Tep) {
    int search = 0;

    while (search <= max_restarts_) {
        int iter = 1;
        while (iter <= max_iter_) {
            compute_fk_and_jacob0();
            angle_axis(Te_, Tep, e_);

            if (we_is_identity_) {
                residual_ = 0.5 * e_.squaredNorm();
            } else {
                residual_ = 0.5 * (e_.transpose() * We_ * e_)(0, 0);
            }

            if (residual_ < tol_) {
                wrap_to_limits();
                if (enforce_limits_) {
                    success_ = check_limits();
                } else {
                    success_ = true;
                }
                iterations_ += iter;
                break;
            }

            if (we_is_identity_) {
                g_.noalias() = J_.transpose() * e_;
                JtWJ_.noalias() = J_.transpose() * J_;
            } else {
                g_.noalias() = J_.transpose() * (We_ * e_);
                JtWJ_.noalias() = J_.transpose() * We_ * J_;
            }

            q_ += JtWJ_.colPivHouseholderQr().solve(g_);
            if (nullspace_active_) {
                calc_qnull();
                q_ += qnull_;
            }

            iter += 1;
        }

        if (success_) {
            break;
        }

        iterations_ += iter;
        search += 1;
        restarts_ = search;
        rand_q();
    }
}

// ===== Newton-Raphson (ported from _IK_NR in ik.cpp) =====
void IKSolver::solve_nr(const Eigen::Matrix4d& Tep) {
    int search = 0;

    while (search <= max_restarts_) {
        int iter = 1;
        while (iter <= max_iter_) {
            compute_fk_and_jacob0();
            angle_axis(Te_, Tep, e_);

            if (we_is_identity_) {
                residual_ = 0.5 * e_.squaredNorm();
            } else {
                residual_ = 0.5 * (e_.transpose() * We_ * e_)(0, 0);
            }

            if (residual_ < tol_) {
                wrap_to_limits();
                if (enforce_limits_) {
                    success_ = check_limits();
                } else {
                    success_ = true;
                }
                iterations_ += iter;
                break;
            }

            // For non-square Jacobians, use pseudo-inverse via SVD
            if (robot_.nq() != 6) {
                Eigen::JacobiSVD<Eigen::MatrixXd> svd(
                    J_, Eigen::ComputeThinU | Eigen::ComputeThinV);
                q_ += svd.solve(e_);
            } else {
                // Square case: direct solve
                q_ += J_.colPivHouseholderQr().solve(e_);
            }
            if (nullspace_active_) {
                calc_qnull();
                q_ += qnull_;
            }

            iter += 1;
        }

        if (success_) {
            break;
        }

        iterations_ += iter;
        search += 1;
        restarts_ = search;
        rand_q();
    }
}

// ===== Levenberg-Marquardt (all 3 damping variants) =====
void IKSolver::solve_lm(const Eigen::Matrix4d& Tep) {
    int search = 0;

    while (search <= max_restarts_) {
        int iter = 1;
        while (iter <= max_iter_) {
            compute_fk_and_jacob0();
            angle_axis(Te_, Tep, e_);

            if (we_is_identity_) {
                residual_ = 0.5 * e_.squaredNorm();
            } else {
                residual_ = 0.5 * (e_.transpose() * We_ * e_)(0, 0);
            }

            if (residual_ < tol_) {
                wrap_to_limits();
                if (enforce_limits_) {
                    success_ = check_limits();
                } else {
                    success_ = true;
                }
                iterations_ += iter;
                break;
            }

            // Compute damping scalar
            double wn;
            switch (damping_) {
                case Damping::Chan:
                    wn = lambda_ * residual_;
                    break;
                case Damping::Wampler:
                    wn = lambda_;
                    break;
                case Damping::Sugihara:
                    wn = residual_ + lambda_;
                    break;
            }

            // Build J^T * We * J + wn * I and gradient
            if (we_is_identity_) {
                g_.noalias() = J_.transpose() * e_;
                JtWJ_.noalias() = J_.transpose() * J_;
            } else {
                g_.noalias() = J_.transpose() * (We_ * e_);
                JtWJ_.noalias() = J_.transpose() * We_ * J_;
            }
            // Add damping to diagonal (avoids allocating Wn_ matrix)
            JtWJ_.diagonal().array() += wn;

            q_ += JtWJ_.colPivHouseholderQr().solve(g_);
            if (nullspace_active_) {
                calc_qnull();
                q_ += qnull_;
            }

            iter += 1;
        }

        if (success_) {
            break;
        }

        iterations_ += iter;
        search += 1;
        restarts_ = search;
        rand_q();
    }
}

// ===== Wrap solution angles to stay within joint limits =====
// For each joint, tries q, q±2π, and wrapToPi(q). Among all variants that
// fall within limits, picks the one closest to q0_ (the seed) to minimize
// unnecessary joint motion. This replaces the old separate unwrap_angles step.
void IKSolver::wrap_to_limits() {
    const auto& ql = robot_.lower_limits();
    const auto& qh = robot_.upper_limits();

    for (int i = 0; i < robot_.nq(); i++) {
        double q_orig = q_(i);
        double ql_min = ql(i);
        double ql_max = qh(i);
        double q0i = q0_(i);

        // Candidates: original, ±2π, wrapToPi
        double candidates[4] = {q_orig, q_orig + PI_x2, q_orig - PI_x2, wrapToPi(q_orig)};

        double best = q_orig;
        double best_dist = std::numeric_limits<double>::max();
        bool found = false;

        for (int c = 0; c < 4; c++) {
            if (candidates[c] >= ql_min && candidates[c] <= ql_max) {
                double dist = std::abs(candidates[c] - q0i);
                if (dist < best_dist) {
                    best_dist = dist;
                    best = candidates[c];
                    found = true;
                }
            }
        }

        if (found) {
            q_(i) = best;
        }
    }
}

// ===== Check joint limits =====
// Ported from ik.cpp _check_lim
bool IKSolver::check_limits() const {
    const auto& ql = robot_.lower_limits();
    const auto& qh = robot_.upper_limits();

    for (int i = 0; i < robot_.nq(); i++) {
        if (q_(i) < ql(i) || q_(i) > qh(i)) {
            return false;
        }
    }
    return true;
}

// ===== Random q within joint limits =====
// Ported from ik.cpp _rand_q
void IKSolver::rand_q() {
    const auto& ql = robot_.lower_limits();
    const auto& qh = robot_.upper_limits();
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < robot_.nq(); i++) {
        q_(i) = ql(i) + dist(rng_) * (qh(i) - ql(i));
    }
}

// ===== Fused FK + Jacobian (single Pinocchio pass) =====
void IKSolver::compute_fk_and_jacob0() {
    const auto& model = robot_.model();
    auto& data = robot_.data();
    auto frame_id = robot_.ee_frame_id();

    // One pass: forwardKinematics + computeJointJacobians
    pinocchio::computeJointJacobians(model, data, q_);
    pinocchio::updateFramePlacement(model, data, frame_id);

    // Extract FK
    Te_ = data.oMf[frame_id].toHomogeneousMatrix();

    // Extract Jacobian (no recomputation — uses already-computed data)
    J_.setZero();
    pinocchio::getFrameJacobian(model, data, frame_id,
                                pinocchio::LOCAL_WORLD_ALIGNED, J_);

    // Tool correction (if active)
    if (robot_.has_tool_transform()) {
        // FK: post-multiply tool transform
        Te_ = Te_ * robot_.tool_transform();
        // Jacobian: v_tool = v_ee + omega x (R_ee * p_tool)
        Eigen::Vector3d r = data.oMf[frame_id].rotation() * robot_.tool_offset();
        skew_r_ <<     0, -r(2),  r(1),
                    r(2),     0, -r(0),
                   -r(1),  r(0),     0;
        J_.topRows(3) -= skew_r_ * J_.bottomRows(3);
    }
}

// ===== angle_axis error function =====
// Ported verbatim from RTB methods.cpp:673-718
void IKSolver::angle_axis(const Eigen::Matrix4d& Te,
                          const Eigen::Matrix4d& Tep,
                          Eigen::Matrix<double, 6, 1>& e) {
    // e[:3] = Tep[:3, 3] - Te[:3, 3]
    e.head<3>() = Tep.block<3, 1>(0, 3) - Te.block<3, 1>(0, 3);

    // R = Tep[:3, :3] @ Te[:3, :3].T
    Eigen::Matrix3d R = Tep.block<3, 3>(0, 0) * Te.block<3, 3>(0, 0).transpose();

    // li = [R(2,1) - R(1,2), R(0,2) - R(2,0), R(1,0) - R(0,1)]
    Eigen::Vector3d li;
    li << R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1);

    double li_norm = li.norm();
    double R_tr = R.trace();

    if (li_norm < 1e-12) {
        // Diagonal matrix case
        if (R_tr > 0) {
            // (1,1,1) case - zero rotation error
            e.tail<3>().setZero();
        } else {
            // 180-degree rotation case
            e(3) = PI_2 * (R(0, 0) + 1);
            e(4) = PI_2 * (R(1, 1) + 1);
            e(5) = PI_2 * (R(2, 2) + 1);
        }
    } else {
        // General case
        double ang = std::atan2(li_norm, R_tr - 1);
        e.tail<3>() = ang * li / li_norm;
    }
}

// ===== Batch IK =====
IKSolver::BatchResult IKSolver::batch_ik(
    const std::vector<Eigen::Matrix4d>& poses,
    const Eigen::VectorXd& q_start,
    bool stop_on_failure) {

    const int n_poses = static_cast<int>(poses.size());
    const int n = robot_.nq();

    BatchResult result;
    result.joint_positions.resize(n_poses, n);
    result.valid.resize(n_poses, false);
    result.all_valid = true;

    Eigen::VectorXd q_warm = q_start;

    for (int i = 0; i < n_poses; ++i) {
        bool ok = solve(poses[i], &q_warm);
        result.valid[i] = ok;

        if (ok) {
            result.joint_positions.row(i) = q_;
            q_warm = q_;
        } else {
            result.joint_positions.row(i).setZero();
            result.all_valid = false;
            if (stop_on_failure) {
                // Zero + invalidate remaining rows and break
                for (int j = i + 1; j < n_poses; ++j) {
                    result.joint_positions.row(j).setZero();
                    // valid[j] already false from resize
                }
                break;
            }
            // Keep q_warm from last successful solve
        }
    }

    return result;
}

// ===== Null-space optimization (ported from RTB IK.py _null_Σ / _calc_qnull) =====
//
// Σ_i = -((qi - ql_min) - pi)² / (ps - pi)²   if qi - ql_min ≤ pi
// Σ_i = +((ql_max - qi) - pi)² / (ps - pi)²   if ql_max - qi ≤ pi
// Σ_i = 0                                     otherwise
//
// RTB returns `-Σ`, producing a gradient that pushes joints away from limits.
void IKSolver::null_sigma(Eigen::VectorXd& sigma_out) const {
    const auto& ql = robot_.lower_limits();
    const auto& qh = robot_.upper_limits();
    const int n = robot_.nq();

    const double denom = (ps_ - pi_);
    const double denom_sq = denom * denom;

    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        const double lower_gap = q_(i) - ql(i);
        const double upper_gap = qh(i) - q_(i);
        if (lower_gap <= pi_) {
            const double t = lower_gap - pi_;
            s = -(t * t) / denom_sq;
        }
        if (upper_gap <= pi_) {
            const double t = upper_gap - pi_;
            s = (t * t) / denom_sq;
        }
        // RTB returns -Σ so the gradient points away from the violated limit
        sigma_out(i) = -s;
    }
}

// Manipulability gradient Jm_i = ∂m/∂q_i where m = sqrt(det(J Jᵀ)).
// Central finite difference on jacob0 — cost is 2n Jacobian evaluations.
// Only invoked when km_ > 0.
void IKSolver::jacobm_fd(Eigen::VectorXd& Jm_out) {
    const int n = robot_.nq();
    const double h = 1e-6;

    auto manip_at = [&](const Eigen::MatrixXd& J) {
        // m = sqrt(det(J Jᵀ)); JtJ_ aliased as workspace for 6x6 JJᵀ
        Eigen::Matrix<double, 6, 6> JJt = J * J.transpose();
        double d = JJt.determinant();
        return (d > 0.0) ? std::sqrt(d) : 0.0;
    };

    q_perturb_ = q_;
    for (int i = 0; i < n; ++i) {
        q_perturb_(i) = q_(i) + h;
        robot_.jacob0(q_perturb_, J_fd_plus_);
        q_perturb_(i) = q_(i) - h;
        robot_.jacob0(q_perturb_, J_fd_minus_);
        q_perturb_(i) = q_(i);

        const double m_plus = manip_at(J_fd_plus_);
        const double m_minus = manip_at(J_fd_minus_);
        Jm_out(i) = (m_plus - m_minus) / (2.0 * h);
    }
}

// Computes the null-space correction qnull_ = (I - J⁺ J) · grad,
// where grad combines the joint-limit and manipulability gradients.
// For a full-rank square J, (I - J⁺ J) ≈ 0 and qnull_ vanishes — that
// is correct behavior for a 6-DOF non-redundant arm with a full 6-D task.
void IKSolver::calc_qnull() {
    if (!nullspace_active_) {
        qnull_.setZero();
        return;
    }

    qnull_grad_.setZero();

    if (kq_ > 0.0) {
        null_sigma(sigma_);
        qnull_grad_.noalias() += (1.0 / kq_) * sigma_;
    }

    if (km_ > 0.0) {
        jacobm_fd(Jm_);
        qnull_grad_.noalias() += (1.0 / km_) * Jm_;
    }

    // Pseudoinverse via SVD for rank-robust null-space projector.
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(J_, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Jpinv_.noalias() = svd.solve(
        Eigen::MatrixXd::Identity(J_.rows(), J_.rows()));

    const int n = robot_.nq();
    N_.noalias() = Eigen::MatrixXd::Identity(n, n) - Jpinv_ * J_;
    qnull_.noalias() = N_ * qnull_grad_;
}

} // namespace pinokin
