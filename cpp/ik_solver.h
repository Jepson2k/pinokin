#pragma once

#include "robot.h"
#include <Eigen/Dense>
#include <vector>
#include <random>

namespace pinokin {

class IKSolver {
public:
    enum class Method { GN, NR, LM };
    enum class Damping { Chan, Wampler, Sugihara };

    IKSolver(const Robot& robot, Method method = Method::LM,
             Damping damping = Damping::Sugihara,
             double tol = 1e-6, double lambda = 1.0,
             int max_iter = 30, int max_restarts = 100,
             bool enforce_limits = true);

    bool solve(const Eigen::Matrix4d& Tep,
               const Eigen::VectorXd* q0 = nullptr);

    struct BatchResult {
        Eigen::MatrixXd joint_positions;  // N x nq
        std::vector<bool> valid;
        bool all_valid;
    };

    BatchResult batch_ik(const std::vector<Eigen::Matrix4d>& poses,
                         const Eigen::VectorXd& q_start);

    static Eigen::VectorXd unwrap_angles(const Eigen::VectorXd& q_solution,
                                         const Eigen::VectorXd& q_current);

    // Results (valid after solve())
    const Eigen::VectorXd& q() const { return q_; }
    bool success() const { return success_; }
    double residual() const { return residual_; }
    int iterations() const { return iterations_; }
    int restarts() const { return restarts_; }

    void set_we(const Eigen::VectorXd& we);

private:
    const Robot& robot_;
    Method method_;
    Damping damping_;
    double tol_;
    double lambda_;
    int max_iter_;
    int max_restarts_;
    bool enforce_limits_;
    bool we_is_identity_;  // fast path when We_ == I
    Eigen::Matrix<double, 6, 6> We_;

    // Pre-allocated workspace (fixed-size for 6-DOF fast path)
    Eigen::Matrix<double, 6, Eigen::Dynamic> J_;   // 6 x nq
    Eigen::Matrix<double, 6, 1> e_;                 // 6
    Eigen::MatrixXd JtWJ_;   // nq x nq
    Eigen::VectorXd g_;      // nq
    Eigen::Matrix4d Te_;     // current FK

    // Results
    Eigen::VectorXd q_;
    bool success_;
    double residual_;
    int iterations_;
    int restarts_;

    // RNG for random restarts
    std::mt19937 rng_;

    // Solver internals - ported from RTB ik.cpp
    void solve_gn(const Eigen::Matrix4d& Tep);
    void solve_nr(const Eigen::Matrix4d& Tep);
    void solve_lm(const Eigen::Matrix4d& Tep);

    void wrap_to_limits();
    bool check_limits() const;
    void rand_q();

    // Fused FK + Jacobian: single forwardKinematics pass, one frame update
    void compute_fk_and_jacob0();

    // Ported from RTB methods.cpp:673-718
    static void angle_axis(const Eigen::Matrix4d& Te,
                           const Eigen::Matrix4d& Tep,
                           Eigen::Matrix<double, 6, 1>& e);

    // Cached skew matrix for tool correction (avoids per-iter allocation)
    Eigen::Matrix3d skew_r_;
};

} // namespace pinokin
