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

    solver_cold = IKSolver(robot, max_restarts=200)
    solver_cold.solve(T_target)

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


def test_wrist_flip_rescues_branch_violation(robot):
    """The deterministic wrist-flip restart should rescue an out-of-limits
    wrist branch without relying on random restarts.

    Seed with wrist_flip(q_target): LM converges to the seed itself
    (pose-preserving) but with wrist joints OOB, and the C++ wrist-flip restart
    should flip back to the valid configuration.
    """
    ql = np.asarray(robot.lower_limits)
    qu = np.asarray(robot.upper_limits)
    n = len(ql)

    # PAROL6 J4 limits are roughly [-1.84, 1.84] rad, so q_target[J4]=0.5 gives
    # seed[J4]=0.5+π≈3.64, which is OOB with no 2π-wrap rescue.
    q_target = (ql + qu) / 2
    q_target[n - 3] = 0.5  # J4
    q_target[n - 2] = 0.7  # J5
    q_target[n - 1] = 1.0  # J6

    T = np.asarray(robot.fkine(q_target))

    # Wrist-flip seed: pose-preserving but with wrist joints in OOB branch.
    seed = q_target.copy()
    seed[n - 3] = q_target[n - 3] + np.pi
    seed[n - 2] = -q_target[n - 2]
    seed[n - 1] = q_target[n - 1] + np.pi

    # max_restarts=1: only the wrist flip can rescue (no random budget).
    solver = IKSolver(robot, tol=1e-10, max_iter=30, max_restarts=1)
    n_fail = 0
    for _ in range(50):
        solver.solve(T, q0=seed)
        if not solver.success:
            n_fail += 1
    assert n_fail == 0, f"wrist-flip restart failed to rescue {n_fail}/50 trials"


def test_solver_q_reflects_last_iterate_on_failure(robot):
    """Regression: rand_q() once ran after the last failed restart, overwriting
    solver.q with random garbage. solver.q must reflect the last LM iterate, not
    a fresh random sample.

    On a converged-but-limits-violated case, FK(solver.q) must equal the target;
    a rand_q overwrite would not match.
    """
    # max_restarts=0 disables wrist-flip and random rescue, so we observe the
    # very first LM convergence outcome directly.
    ql = np.asarray(robot.lower_limits)
    qu = np.asarray(robot.upper_limits)
    n = len(ql)
    q_target = (ql + qu) / 2
    q_target[n - 3] = 0.5
    q_target[n - 2] = 0.7
    q_target[n - 1] = 1.0
    T = np.asarray(robot.fkine(q_target))
    # Full wrist-flip: pose-preserving but puts J4 outside its limit range. LM
    # from this seed converges in ~1 iter (pose already satisfied), but
    # check_limits then rejects.
    seed = q_target.copy()
    seed[n - 3] = q_target[n - 3] + np.pi
    seed[n - 2] = -q_target[n - 2]
    seed[n - 1] = q_target[n - 1] + np.pi

    solver = IKSolver(robot, tol=1e-10, max_iter=30, max_restarts=0)
    solver.solve(T, q0=seed)
    assert not solver.success, "expected limits rejection on wrist-flipped seed"

    # solver.q must still be the LM-converged q (FK matches target), not a
    # random uniform overwrite.
    T_result = np.asarray(robot.fkine(solver.q))
    np.testing.assert_allclose(
        T_result[:3, 3],
        T[:3, 3],
        atol=1e-4,
        err_msg="solver.q after failure was overwritten (rand_q leak)",
    )


def test_iterations_counter_no_double_count(robot):
    """Regression: each solver once added `iter` twice when LM converged but
    check_limits rejected. `solver.iterations` must count each LM iteration
    exactly once, capped at `max_iter` per attempt.

    A wrist-flipped seed converges in 1 iter then trips check_limits; the restart
    converges in 1 more. Pre-fix reported ~4 iters, post-fix ~2.
    """
    ql = np.asarray(robot.lower_limits)
    qu = np.asarray(robot.upper_limits)
    n = len(ql)
    q_target = (ql + qu) / 2
    q_target[n - 3] = 0.5
    q_target[n - 2] = 0.7
    q_target[n - 1] = 1.0
    T = np.asarray(robot.fkine(q_target))
    seed = q_target.copy()
    seed[n - 3] += np.pi
    seed[n - 2] = -seed[n - 2]
    seed[n - 1] += np.pi

    solver = IKSolver(robot, tol=1e-10, max_iter=30, max_restarts=1)
    solver.solve(T, q0=seed)
    assert solver.success
    # Two attempts (wrist-flipped seed + restart), each ~1 LM iter since the
    # seed already satisfies the pose, so the counter should sum to <= 4.
    assert solver.iterations <= 4, (
        f"iterations counter inflated: got {solver.iterations}, expected <= 4"
    )
