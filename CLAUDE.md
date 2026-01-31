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

- `cpp/robot.h/.cpp` — Robot class wrapping Pinocchio (FK, Jacobians, batch FK)
- `cpp/ik_solver.h/.cpp` — IK solver (GN, NR, LM with 3 damping variants)
- `cpp/bindings.cpp` — nanobind Python bindings
- `src/pinokin/__init__.py` — Python package entry point
- `src/pinokin/_core.pyi` — type stubs for IDE support
- `data/parol6.urdf` — test URDF (PAROL6 robot)

## C++ namespace

All C++ code lives in `namespace pinokin`.

## Key conventions

- FK returns 4x4 numpy arrays (Eigen::Matrix4d)
- Jacobians are 6xN with [linear; angular] row ordering
- jacob0 uses LOCAL_WORLD_ALIGNED (world-frame orientation)
- jacobe uses LOCAL (end-effector frame)
- IKSolver owns pre-allocated workspace — reuse instances across calls
