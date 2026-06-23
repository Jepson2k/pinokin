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
                   bool enforce_limits)
    : robot_(robot)
    , method_(method)
    , damping_(damping)
    , tol_(tol)
    , lambda_(lambda)
    , max_iter_(max_iter)
    , max_restarts_(max_restarts)
    , enforce_limits_(enforce_limits)
    , we_is_identity_(true)
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

    detect_spherical_wrist();
}

void IKSolver::set_we(const Eigen::VectorXd& we) {
    if (we.size() != 6) {
        throw std::runtime_error("we must be a 6-vector");
    }
    We_ = we.asDiagonal();
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

// Gauss-Newton (ported from _IK_GN in ik.cpp).
void IKSolver::solve_gn(const Eigen::Matrix4d& Tep) {
    int search = 0;
    bool wrist_flip_tried = false;

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

            iter += 1;
        }

        // Count attempt iterations once: if inner loop broke at convergence,
        // iter is the converging step (1-indexed); if it exhausted naturally,
        // iter == max_iter_+1 and we cap to max_iter_.
        iterations_ += (iter > max_iter_ ? max_iter_ : iter);

        if (success_) {
            break;
        }

        search += 1;
        restarts_ = search;
        if (search <= max_restarts_) {
            if (!wrist_flip_tried && wrist_only_violations()) {
                apply_wrist_flip();
                wrist_flip_tried = true;
            } else {
                rand_q();
            }
        }
    }
}

// Newton-Raphson (ported from _IK_NR in ik.cpp).
void IKSolver::solve_nr(const Eigen::Matrix4d& Tep) {
    int search = 0;
    bool wrist_flip_tried = false;

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
                break;
            }

            // For non-square Jacobians, use pseudo-inverse via SVD
            if (robot_.nq() != 6) {
                Eigen::JacobiSVD<Eigen::MatrixXd> svd(
                    J_, Eigen::ComputeThinU | Eigen::ComputeThinV);
                q_ += svd.solve(e_);
            } else {
                q_ += J_.colPivHouseholderQr().solve(e_);
            }

            iter += 1;
        }

        iterations_ += (iter > max_iter_ ? max_iter_ : iter);

        if (success_) {
            break;
        }

        search += 1;
        restarts_ = search;
        if (search <= max_restarts_) {
            if (!wrist_flip_tried && wrist_only_violations()) {
                apply_wrist_flip();
                wrist_flip_tried = true;
            } else {
                rand_q();
            }
        }
    }
}

// Levenberg-Marquardt (all 3 damping variants).
void IKSolver::solve_lm(const Eigen::Matrix4d& Tep) {
    int search = 0;
    bool wrist_flip_tried = false;

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
                break;
            }

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

            if (we_is_identity_) {
                g_.noalias() = J_.transpose() * e_;
                JtWJ_.noalias() = J_.transpose() * J_;
            } else {
                g_.noalias() = J_.transpose() * (We_ * e_);
                JtWJ_.noalias() = J_.transpose() * We_ * J_;
            }
            // Damp the diagonal in place to avoid allocating a Wn_ matrix.
            JtWJ_.diagonal().array() += wn;

            q_ += JtWJ_.colPivHouseholderQr().solve(g_);

            iter += 1;
        }

        iterations_ += (iter > max_iter_ ? max_iter_ : iter);

        if (success_) {
            break;
        }

        search += 1;
        restarts_ = search;
        if (search <= max_restarts_) {
            // First restart on a wrist-only-violation: try the deterministic
            // wrist flip before falling back to random restarts. This finds
            // the kinematically-equivalent IK branch in joint space without
            // a stochastic search.
            if (!wrist_flip_tried && wrist_only_violations()) {
                apply_wrist_flip();
                wrist_flip_tried = true;
            } else {
                rand_q();
            }
        }
    }
}

// Wrap solution angles to stay within joint limits.
// For each joint, tries q, q±2π, and wrapToPi(q). Among all variants that
// fall within limits, picks the one closest to q0_ (the seed) to minimize
// unnecessary joint motion.
void IKSolver::wrap_to_limits() {
    const auto& ql = robot_.lower_limits();
    const auto& qh = robot_.upper_limits();

    for (int i = 0; i < robot_.nq(); i++) {
        double q_orig = q_(i);
        double ql_min = ql(i);
        double ql_max = qh(i);
        double q0i = q0_(i);

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

// Ported from ik.cpp _check_lim.
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

// Random q within joint limits. Ported from ik.cpp _rand_q.
void IKSolver::rand_q() {
    const auto& ql = robot_.lower_limits();
    const auto& qh = robot_.upper_limits();
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < robot_.nq(); i++) {
        q_(i) = ql(i) + dist(rng_) * (qh(i) - ql(i));
    }
}

// Detect spherical wrist: last 3 joint axes intersect at one point.
// Probes the linear-velocity columns of the world Jacobian at q=0. If all
// three wrist axes pass through a common point (the wrist center), then the
// EE-linear-velocity contributions from each are coplanar (all perpendicular
// to (p_ee - p_wrist_center)), making the 3x3 block rank-deficient.
void IKSolver::detect_spherical_wrist() {
    int n = robot_.nq();
    if (n < 3) {
        has_spherical_wrist_ = false;
        return;
    }
    Eigen::VectorXd q_zero = Eigen::VectorXd::Zero(n);
    Eigen::Matrix<double, 6, Eigen::Dynamic> J(6, n);
    J.setZero();
    // Manually invoke FK + Jacobian at q=0 (don't disturb solver state)
    auto saved_q = q_;
    q_ = q_zero;
    compute_fk_and_jacob0();
    J = J_;
    q_ = saved_q;

    Eigen::Matrix3d last3 = J.block<3, 3>(0, n - 3);
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(last3);
    // Spherical wrist iff smallest singular value is near zero (rank ≤ 2)
    double smallest = svd.singularValues()(2);
    has_spherical_wrist_ = (smallest < 1e-4);
    wrist_start_ = has_spherical_wrist_ ? (n - 3) : -1;
}

// Are all out-of-limit joints in the wrist?
bool IKSolver::wrist_only_violations() const {
    if (!has_spherical_wrist_) return false;
    const auto& ql = robot_.lower_limits();
    const auto& qh = robot_.upper_limits();
    bool any_violation = false;
    for (int i = 0; i < robot_.nq(); i++) {
        if (q_(i) < ql(i) || q_(i) > qh(i)) {
            if (i < wrist_start_) return false;  // non-wrist violation
            any_violation = true;
        }
    }
    return any_violation;
}

// Wrist flip: q[w]+π, -q[w+1], q[w+2]+π.
// Pose-preserving for any robot with a spherical wrist (last 3 axes intersect).
// Generates the alternative IK solution that exists in joint space when the
// initial LM convergence lands on a wrist branch outside joint limits.
void IKSolver::apply_wrist_flip() {
    int w = wrist_start_;
    q_(w)     += PI;
    q_(w + 1) = -q_(w + 1);
    q_(w + 2) += PI;
}

// Fused FK + Jacobian: a single Pinocchio pass per call.
void IKSolver::compute_fk_and_jacob0() {
    const auto& model = robot_.model();
    auto& data = robot_.data();
    auto frame_id = robot_.ee_frame_id();

    pinocchio::computeJointJacobians(model, data, q_);
    pinocchio::updateFramePlacement(model, data, frame_id);

    Te_ = data.oMf[frame_id].toHomogeneousMatrix();

    // Reuses the data already populated above — no recomputation.
    J_.setZero();
    pinocchio::getFrameJacobian(model, data, frame_id,
                                pinocchio::LOCAL_WORLD_ALIGNED, J_);

    if (robot_.has_tool_transform()) {
        Te_ = Te_ * robot_.tool_transform();
        // Jacobian: v_tool = v_ee + omega x (R_ee * p_tool)
        Eigen::Vector3d r = data.oMf[frame_id].rotation() * robot_.tool_offset();
        skew_r_ <<     0, -r(2),  r(1),
                    r(2),     0, -r(0),
                   -r(1),  r(0),     0;
        J_.topRows(3) -= skew_r_ * J_.bottomRows(3);
    }
}

// angle_axis error function. Ported verbatim from RTB methods.cpp:673-718.
void IKSolver::angle_axis(const Eigen::Matrix4d& Te,
                          const Eigen::Matrix4d& Tep,
                          Eigen::Matrix<double, 6, 1>& e) {
    e.head<3>() = Tep.block<3, 1>(0, 3) - Te.block<3, 1>(0, 3);

    Eigen::Matrix3d R = Tep.block<3, 3>(0, 0) * Te.block<3, 3>(0, 0).transpose();

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

} // namespace pinokin
