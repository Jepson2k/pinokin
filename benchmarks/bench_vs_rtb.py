"""Benchmark pinokin vs roboticstoolbox-python.

Compares FK, Jacobian, IK, batch FK, and import time.
"""

import sys
import time
import timeit
from pathlib import Path

import numpy as np


URDF_PATH = str(Path(__file__).parent.parent / "data" / "parol6.urdf")
URDF_STRING = Path(URDF_PATH).read_text()

N_FK = 5000
N_JAC = 5000
N_IK = 100
N_BATCH = 500


def bench_import():
    """Measure import time for both libraries."""
    results = {}

    # pinokin
    t0 = time.perf_counter()
    import pinokin  # noqa: F401
    results["pinokin"] = time.perf_counter() - t0

    # RTB — may not be installed
    try:
        # Force reimport by clearing cache
        mods_before = set(sys.modules.keys())
        t0 = time.perf_counter()
        import roboticstoolbox  # noqa: F401
        results["rtb"] = time.perf_counter() - t0
    except ImportError:
        results["rtb"] = None

    return results


def make_random_configs(robot_nq, lower, upper, n, rng):
    configs = np.zeros((n, robot_nq))
    for i in range(n):
        configs[i] = lower + rng.random(robot_nq) * (upper - lower)
    return configs


def bench_pinokin():
    from pinokin import Robot, IKSolver

    robot = Robot(URDF_PATH)
    rng = np.random.default_rng(42)
    ql, qu = robot.lower_limits, robot.upper_limits
    configs = make_random_configs(robot.nq, ql, qu, max(N_FK, N_JAC, N_BATCH), rng)
    q_mid = (ql + qu) / 2

    results = {}

    # FK
    def run_fk():
        for i in range(N_FK):
            robot.fkine(configs[i % len(configs)])
    results["fk"] = timeit.timeit(run_fk, number=1)

    # Jacobian
    def run_jac():
        for i in range(N_JAC):
            robot.jacob0(configs[i % len(configs)])
    results["jacobian"] = timeit.timeit(run_jac, number=1)

    # Batch FK
    batch_configs = configs[:N_BATCH]
    def run_batch_fk():
        robot.batch_fk(batch_configs)
    results["batch_fk"] = timeit.timeit(run_batch_fk, number=1)

    # IK
    targets = []
    for i in range(N_IK):
        targets.append(robot.fkine(configs[i]))
    solver = IKSolver(robot, tol=1e-12, max_iter=10, max_restarts=10)

    def run_ik():
        for i in range(N_IK):
            solver.solve(targets[i], q0=q_mid)
    results["ik"] = timeit.timeit(run_ik, number=1)

    return results


def bench_rtb():
    try:
        import roboticstoolbox as rtb
        from roboticstoolbox.fknm import ETS_fkine
    except ImportError:
        return None

    from roboticstoolbox.tools.urdf import URDF

    links, name, urdf_string, file_path = URDF.loadstr(
        URDF_STRING, tld="", xacro_tld=""
    )
    robot = rtb.Robot(links, name=name)
    ets = robot.ets()
    fknm = ets._fknm

    rng = np.random.default_rng(42)
    ql = robot.qlim[0]
    qu = robot.qlim[1]
    configs = make_random_configs(robot.n, ql, qu, max(N_FK, N_JAC, N_BATCH), rng)
    q_mid = (ql + qu) / 2

    results = {}

    # FK (raw C path)
    T_out = np.asfortranarray(np.eye(4))
    def run_fk():
        for i in range(N_FK):
            ETS_fkine(fknm, configs[i % len(configs)], None, None, True, T_out)
    results["fk"] = timeit.timeit(run_fk, number=1)

    # Jacobian
    def run_jac():
        for i in range(N_JAC):
            robot.jacob0(configs[i % len(configs)])
    results["jacobian"] = timeit.timeit(run_jac, number=1)

    # Batch FK (Python loop — RTB has no batch_fk)
    def run_batch_fk():
        for i in range(N_BATCH):
            ETS_fkine(fknm, configs[i], None, None, True, T_out)
    results["batch_fk"] = timeit.timeit(run_batch_fk, number=1)

    # IK
    from roboticstoolbox.robot.IK import IKResultBuffer
    targets = []
    for i in range(N_IK):
        ETS_fkine(fknm, configs[i], None, None, True, T_out)
        targets.append(T_out.copy())

    result_buf = IKResultBuffer(ets.n)

    def run_ik():
        for i in range(N_IK):
            ets.ik_LM(
                targets[i], q0=q_mid, tol=1e-12, joint_limits=True,
                k=0.0, method="sugihara", ilimit=10, slimit=10,
                result=result_buf,
            )
    results["ik"] = timeit.timeit(run_ik, number=1)

    return results


def format_time(seconds, n_ops):
    """Format as time per operation."""
    us_per_op = (seconds / n_ops) * 1e6
    if us_per_op >= 1000:
        return f"{us_per_op / 1000:.2f} ms"
    return f"{us_per_op:.1f} us"


def main():
    print("=" * 70)
    print("pinokin vs roboticstoolbox-python benchmark")
    print("=" * 70)
    print()

    # Import benchmark
    print("--- Import time ---")
    imp = bench_import()
    print(f"  pinokin:  {imp['pinokin']*1000:.0f} ms")
    if imp["rtb"] is not None:
        print(f"  RTB:      {imp['rtb']*1000:.0f} ms")
    else:
        print("  RTB:      not installed")
    print()

    # pinokin benchmarks
    print(f"--- pinokin ({N_FK} FK, {N_JAC} Jac, {N_BATCH} batch FK, {N_IK} IK) ---")
    pk = bench_pinokin()
    print(f"  FK:       {format_time(pk['fk'], N_FK)}/call  (total {pk['fk']:.3f}s)")
    print(f"  Jacobian: {format_time(pk['jacobian'], N_JAC)}/call  (total {pk['jacobian']:.3f}s)")
    print(f"  Batch FK: {format_time(pk['batch_fk'], N_BATCH)}/call  (total {pk['batch_fk']:.3f}s)")
    print(f"  IK:       {format_time(pk['ik'], N_IK)}/call  (total {pk['ik']:.3f}s)")
    print()

    # RTB benchmarks
    rtb_results = bench_rtb()
    if rtb_results is not None:
        print(f"--- RTB ({N_FK} FK, {N_JAC} Jac, {N_BATCH} batch FK, {N_IK} IK) ---")
        print(f"  FK:       {format_time(rtb_results['fk'], N_FK)}/call  (total {rtb_results['fk']:.3f}s)")
        print(f"  Jacobian: {format_time(rtb_results['jacobian'], N_JAC)}/call  (total {rtb_results['jacobian']:.3f}s)")
        print(f"  Batch FK: {format_time(rtb_results['batch_fk'], N_BATCH)}/call  (total {rtb_results['batch_fk']:.3f}s)")
        print(f"  IK:       {format_time(rtb_results['ik'], N_IK)}/call  (total {rtb_results['ik']:.3f}s)")
        print()

        # Comparison
        print("--- Speedup (pinokin / RTB) ---")
        for key in ["fk", "jacobian", "batch_fk", "ik"]:
            ratio = rtb_results[key] / pk[key]
            label = key.upper().replace("_", " ")
            print(f"  {label:12s}: {ratio:.2f}x {'faster' if ratio > 1 else 'slower'}")
    else:
        print("RTB not installed — skipping comparison.")

    print()


if __name__ == "__main__":
    main()
