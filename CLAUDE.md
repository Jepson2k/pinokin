# pinokin — Development Guide

## Development Setup

```bash
conda env create -f environment.yml
conda activate pinokin
pip install -e ".[dev]" --no-build-isolation
```

## Test

```bash
pytest tests/ -v
```

## Linting & Formatting

```bash
ruff check src/pinokin/
ruff format src/pinokin/
mypy src/pinokin/

# Run all pre-commit hooks
pre-commit run --all-files
```

## Building a Distributable Wheel

The editable install (`-e`) links to source but doesn't bundle dependencies.
To build a standalone wheel that works without conda/pinocchio installed:

```bash
cd /path/to/pinokin

# Activate conda env (source the profile first if conda isn't in PATH)
source ~/miniforge3/etc/profile.d/conda.sh  # if needed
conda activate pinokin  # or whatever env name from environment.yml

# 1. Build raw wheel
pip wheel . --no-build-isolation --no-deps --wheel-dir raw-dist/

# 2. Install repair tool
pip install auditwheel patchelf  # Linux
# pip install delocate            # macOS
# pip install delvewheel          # Windows

# 3. Repair wheel (bundles libpinocchio, libboost, etc.)
mkdir -p dist

# Linux (on newer systems, --plat flag is usually required):
LD_LIBRARY_PATH="$CONDA_PREFIX/lib" auditwheel repair -w dist/ --plat manylinux_2_41_aarch64 raw-dist/*.whl

# macOS:
# DYLD_LIBRARY_PATH="$CONDA_PREFIX/lib" delocate-wheel -w dist/ -v raw-dist/*.whl

# Windows:
# python -m delvewheel repair --add-path "$CONDA_PREFIX/Library/bin" -w dist/ raw-dist/*.whl

# 4. Install in target venv
deactivate  # exit conda env
source /path/to/project/.venv/bin/activate
pip install dist/pinokin-*.whl --force-reinstall
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
