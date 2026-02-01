# pinokin — Development Guide

## Build

```bash
conda env create -f environment.yml
conda activate pinokin
pip install -e ".[dev]" --no-build-isolation
```

## Test

```bash
pytest tests/ -v
```

## Architecture

- `cpp/robot.h/.cpp` — Robot class wrapping Pinocchio (FK, Jacobians, batch FK, in-place variants)
- `cpp/ik_solver.h/.cpp` — IK solver (GN, NR, LM with 3 damping variants, fused FK+Jacobian pass)
- `cpp/bindings.cpp` — nanobind Python bindings
- `src/pinokin/__init__.py` — Python package entry point
- `src/pinokin/_core.pyi` — type stubs for IDE support
- `tests/parol6.urdf` — test URDF (PAROL6 robot)

## C++ namespace

All C++ code lives in `namespace pinokin`.

## Key conventions

- FK returns 4x4 numpy arrays (Eigen::Matrix4d)
- `fkine_into` / `jacob0_into` write to pre-allocated buffers (zero-allocation hot path)
- `qlim` returns 2xN array (row 0: lower, row 1: upper)
- Jacobians are 6xN with [linear; angular] row ordering
- jacob0 uses LOCAL_WORLD_ALIGNED (world-frame orientation)
- jacobe uses LOCAL (end-effector frame)
- IKSolver owns pre-allocated workspace — reuse instances across calls
- IK loop uses fused `compute_fk_and_jacob0()` for a single Pinocchio pass per iteration

## Release workflow

On GitHub Release creation (tag `vX.Y.Z`):
1. `wheels.yml` syncs the tag version into `pyproject.toml`
2. Builds wheels for 5 platforms x 4 Python versions
3. Builds sdist
4. Uploads all to the GitHub Release
5. Commits version bump back to `main`
