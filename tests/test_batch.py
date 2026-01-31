import numpy as np
import pytest

from pinokin import Robot, IKSolver


def test_batch_fk_matches_individual(robot):
    """batch_fk results should match individual fkine calls."""
    rng = np.random.default_rng(42)
    ql = robot.lower_limits
    qu = robot.upper_limits

    N = 10
    joint_positions = np.zeros((N, robot.nq))
    for i in range(N):
        joint_positions[i] = ql + rng.random(robot.nq) * (qu - ql)

    batch_results = robot.batch_fk(joint_positions)
    assert len(batch_results) == N

    for i in range(N):
        T_single = robot.fkine(joint_positions[i])
        np.testing.assert_allclose(batch_results[i], T_single, atol=1e-12)


def test_batch_ik_round_trip(robot):
    """batch_ik should solve each pose, verifiable via FK."""
    rng = np.random.default_rng(99)
    ql = robot.lower_limits
    qu = robot.upper_limits

    N = 5
    q_configs = np.zeros((N, robot.nq))
    poses = []
    for i in range(N):
        q_configs[i] = ql + rng.random(robot.nq) * (qu - ql)
        poses.append(robot.fkine(q_configs[i]))

    q_start = q_configs[0]
    solver = IKSolver(robot, max_restarts=200)
    result = solver.batch_ik(poses, q_start)

    assert result.joint_positions.shape == (N, robot.nq)
    assert len(result.valid) == N

    for i in range(N):
        if result.valid[i]:
            T_check = robot.fkine(result.joint_positions[i])
            np.testing.assert_allclose(
                T_check[:3, 3], poses[i][:3, 3], atol=1e-3)


def test_batch_ik_warm_start_chain(robot):
    """batch_ik warm-starts each solve from the previous solution."""
    ql = robot.lower_limits
    qu = robot.upper_limits
    q_mid = (ql + qu) / 2

    # Small trajectory: perturb only joints 0 and 3 by tiny increments
    poses = []
    for i in range(5):
        q = q_mid.copy()
        q[0] += 0.02 * i
        q[3] += 0.02 * i
        q = np.clip(q, ql, qu)
        poses.append(robot.fkine(q))

    solver = IKSolver(robot, max_restarts=200)
    result = solver.batch_ik(poses, q_mid)
    assert result.all_valid


def test_batch_ik_partial_validity(robot):
    """batch_ik should handle unreachable poses gracefully."""
    ql = robot.lower_limits
    qu = robot.upper_limits
    q_mid = (ql + qu) / 2

    # One reachable, one far away
    T_reachable = robot.fkine(q_mid)
    T_unreachable = np.eye(4)
    T_unreachable[:3, 3] = [100.0, 100.0, 100.0]  # Far away

    solver = IKSolver(robot, max_restarts=5, max_iter=10)
    result = solver.batch_ik([T_reachable, T_unreachable], q_mid)

    assert result.valid[0]
    assert not result.valid[1]
    assert not result.all_valid
