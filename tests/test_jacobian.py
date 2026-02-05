import numpy as np


def _numerical_jacobian(robot, q, eps=1e-8):
    """Finite-difference Jacobian in world frame via FK perturbation."""
    n = robot.nq
    J_num = np.zeros((6, n))
    T0 = robot.fkine(q)

    for i in range(n):
        q_plus = q.copy()
        q_plus[i] += eps
        T_plus = robot.fkine(q_plus)

        # Linear part: position difference
        J_num[:3, i] = (T_plus[:3, 3] - T0[:3, 3]) / eps

        # Angular part: use log of relative rotation
        R0 = T0[:3, :3]
        R_plus = T_plus[:3, :3]
        dR = R_plus @ R0.T
        # Extract angular velocity from dR ≈ I + [w]x * eps
        J_num[3, i] = (dR[2, 1] - dR[1, 2]) / (2 * eps)
        J_num[4, i] = (dR[0, 2] - dR[2, 0]) / (2 * eps)
        J_num[5, i] = (dR[1, 0] - dR[0, 1]) / (2 * eps)

    return J_num


def test_jacob0_shape(robot):
    q = np.zeros(robot.nq)
    J = robot.jacob0(q)
    assert J.shape == (6, 6)


def test_jacob0_vs_numerical(robot):
    """Analytical world-frame Jacobian should match finite-difference."""
    rng = np.random.default_rng(123)
    ql = robot.lower_limits
    qu = robot.upper_limits

    for _ in range(5):
        q = ql + rng.random(robot.nq) * (qu - ql)
        J_analytical = robot.jacob0(q)
        J_numerical = _numerical_jacobian(robot, q)
        np.testing.assert_allclose(J_analytical, J_numerical, atol=1e-5)


def test_jacob0_linear_angular_ordering(robot):
    """Verify [linear; angular] row ordering: top 3 rows should correspond
    to position changes, bottom 3 to orientation changes."""
    q = np.zeros(robot.nq)
    J = robot.jacob0(q)
    T = robot.fkine(q)

    eps = 1e-8
    q_pert = q.copy()
    q_pert[0] += eps
    T_pert = robot.fkine(q_pert)

    # Top row should be linear velocity (position derivative)
    dp = (T_pert[:3, 3] - T[:3, 3]) / eps
    np.testing.assert_allclose(J[:3, 0], dp, atol=1e-5)


def test_jacobe_shape(robot):
    q = np.zeros(robot.nq)
    J = robot.jacobe(q)
    assert J.shape == (6, 6)
