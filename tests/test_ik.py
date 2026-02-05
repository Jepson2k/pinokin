import numpy as np
import pytest

from pinokin import Damping, IKSolver, Method


def _round_trip(robot, solver, q_target):
    """FK -> IK -> FK round-trip test."""
    T_target = robot.fkine(q_target)
    ok = solver.solve(T_target, q_target)
    assert ok, f"IK failed, residual={solver.residual}"
    T_result = robot.fkine(solver.q)
    np.testing.assert_allclose(T_result[:3, 3], T_target[:3, 3], atol=1e-4)
    np.testing.assert_allclose(T_result[:3, :3], T_target[:3, :3], atol=1e-3)


class TestIKMethods:
    """Round-trip FK->IK->FK for all 5 solver configurations."""

    @pytest.fixture
    def q_mid(self, robot):
        """A config in the middle of joint limits."""
        ql = robot.lower_limits
        qu = robot.upper_limits
        return (ql + qu) / 2

    @pytest.mark.parametrize(
        "method,damping",
        [
            (Method.GN, Damping.Sugihara),
            (Method.NR, Damping.Sugihara),
            (Method.LM, Damping.Chan),
            (Method.LM, Damping.Wampler),
            (Method.LM, Damping.Sugihara),
        ],
    )
    def test_round_trip(self, robot, q_mid, method, damping):
        solver = IKSolver(robot, method=method, damping=damping)
        _round_trip(robot, solver, q_mid)


def test_warm_start_converges_faster(robot):
    """Warm-starting from a nearby config should need fewer iterations."""
    ql = robot.lower_limits
    qu = robot.upper_limits
    q_target = (ql + qu) / 2

    T_target = robot.fkine(q_target)

    # Cold start (random)
    solver_cold = IKSolver(robot, max_restarts=200)
    solver_cold.solve(T_target)

    # Warm start from close config
    solver_warm = IKSolver(robot, max_restarts=200)
    q_near = q_target + 0.01 * np.ones(robot.nq)
    solver_warm.solve(T_target, q_near)

    assert solver_warm.success
    assert solver_warm.iterations <= solver_cold.iterations or solver_warm.restarts == 0


def test_joint_limit_enforcement(robot):
    """With enforce_limits=True, solution must respect joint limits."""
    ql = robot.lower_limits
    qu = robot.upper_limits
    q_target = (ql + qu) / 2
    T = robot.fkine(q_target)

    solver = IKSolver(robot, enforce_limits=True)
    ok = solver.solve(T, q_target)
    if ok:
        assert np.all(solver.q >= ql - 1e-10)
        assert np.all(solver.q <= qu + 1e-10)


def test_task_space_weighting(robot):
    """Position-only IK via we=[1,1,1,0,0,0]."""
    ql = robot.lower_limits
    qu = robot.upper_limits
    q_target = (ql + qu) / 2
    T = robot.fkine(q_target)

    solver = IKSolver(robot)
    we = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    solver.set_we(we)
    ok = solver.solve(T, q_target)
    assert ok
    T_result = robot.fkine(solver.q)
    # Position should match even if orientation doesn't
    np.testing.assert_allclose(T_result[:3, 3], T[:3, 3], atol=1e-4)


def test_wrap_to_limits_prefers_q0(robot):
    """wrap_to_limits should prefer the ±2π variant closest to q0."""
    ql = robot.lower_limits
    qu = robot.upper_limits
    q0 = (ql + qu) / 2
    T = robot.fkine(q0)

    solver = IKSolver(robot, tol=1e-10, max_iter=50, max_restarts=10)
    solver.solve(T, q0=q0)
    assert solver.success

    # Solution should be close to q0 (no unnecessary 2π jumps)
    assert np.max(np.abs(solver.q - q0)) < 0.1


def test_solver_result_properties(robot):
    ql = robot.lower_limits
    qu = robot.upper_limits
    q = (ql + qu) / 2
    T = robot.fkine(q)

    solver = IKSolver(robot)
    solver.solve(T, q)
    assert solver.success
    assert solver.residual < 1e-6
    assert solver.iterations > 0
    assert solver.q.shape == (robot.nq,)
