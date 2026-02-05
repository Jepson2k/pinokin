import numpy as np


def test_fk_zero_config(robot):
    q = np.zeros(robot.nq)
    T = robot.fkine(q)
    assert T.shape == (4, 4)
    # Must be valid SE3: bottom row [0,0,0,1]
    np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=1e-12)
    # Rotation part must be orthogonal
    R = T[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


def test_fk_varies_with_q(robot):
    T0 = robot.fkine(np.zeros(robot.nq))
    q1 = np.array([0.5, -1.0, 3.0, 0.3, 0.2, 1.0])
    T1 = robot.fkine(q1)
    # Different configs must give different poses
    assert not np.allclose(T0, T1, atol=1e-6)


def test_fk_se3_properties_random(robot):
    """FK at random configs should always produce valid SE3."""
    rng = np.random.default_rng(42)
    ql = robot.lower_limits
    qu = robot.upper_limits

    for _ in range(10):
        q = ql + rng.random(robot.nq) * (qu - ql)
        T = robot.fkine(q)
        R = T[:3, :3]
        np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


def test_nq(robot):
    assert robot.nq == 6


def test_joint_limits(robot):
    assert robot.lower_limits.shape == (6,)
    assert robot.upper_limits.shape == (6,)
    assert np.all(robot.lower_limits < robot.upper_limits)


def test_tool_transform_fk(robot):
    """Tool transform shifts FK output by the tool offset."""
    q = np.zeros(robot.nq)
    T_no_tool = robot.fkine(q)

    T_tool = np.eye(4)
    T_tool[:3, 3] = [0.0, 0.0, 0.1]  # 100mm tool along z
    robot.set_tool_transform(T_tool)
    assert robot.has_tool_transform

    T_with_tool = robot.fkine(q)
    # Position should differ by the rotated tool offset
    assert not np.allclose(T_no_tool[:3, 3], T_with_tool[:3, 3], atol=1e-6)
    # The difference should be R_ee @ [0, 0, 0.1]
    R_ee = T_no_tool[:3, :3]
    expected_pos = T_no_tool[:3, 3] + R_ee @ np.array([0.0, 0.0, 0.1])
    np.testing.assert_allclose(T_with_tool[:3, 3], expected_pos, atol=1e-10)

    robot.clear_tool_transform()
    assert not robot.has_tool_transform
    T_after_clear = robot.fkine(q)
    np.testing.assert_allclose(T_after_clear, T_no_tool, atol=1e-12)


def test_tool_transform_jacobian(robot):
    """Tool-adjusted Jacobian should match numerical differentiation."""
    T_tool = np.eye(4)
    T_tool[:3, 3] = [0.05, 0.0, 0.1]
    robot.set_tool_transform(T_tool)

    rng = np.random.default_rng(77)
    ql = robot.lower_limits
    qu = robot.upper_limits
    q = ql + rng.random(robot.nq) * (qu - ql)

    J = robot.jacob0(q)

    # Numerical jacobian via FK perturbation
    eps = 1e-8
    T0 = robot.fkine(q)
    J_num = np.zeros((6, robot.nq))
    for i in range(robot.nq):
        q_plus = q.copy()
        q_plus[i] += eps
        T_plus = robot.fkine(q_plus)
        J_num[:3, i] = (T_plus[:3, 3] - T0[:3, 3]) / eps
        R0 = T0[:3, :3]
        R_plus = T_plus[:3, :3]
        dR = R_plus @ R0.T
        J_num[3, i] = (dR[2, 1] - dR[1, 2]) / (2 * eps)
        J_num[4, i] = (dR[0, 2] - dR[2, 0]) / (2 * eps)
        J_num[5, i] = (dR[1, 0] - dR[0, 1]) / (2 * eps)

    np.testing.assert_allclose(J, J_num, atol=1e-5)

    robot.clear_tool_transform()
