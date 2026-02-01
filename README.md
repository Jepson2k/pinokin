# pinokin

The core IK math from [robotics-toolbox-python](https://github.com/petercorke/robotics-toolbox-python) combined with [Pinocchio](https://github.com/stack-of-tasks/pinocchio) FK, Jacobian, and SO3/SE3 math. C++ backend with [nanobind](https://github.com/wjakob/nanobind) Python bindings.

Maintains execution speed and zero allocation with lower import time (~ms vs ~1s for RTB) and a reduced dependency set.

## Install

```bash
pip install pinokin
```

Pre-built wheels are available on [GitHub Releases](https://github.com/Jepson2k/pinokin/releases).

## Usage

```python
from pinokin import Robot, IKSolver
import numpy as np

robot = Robot("path/to/robot.urdf")
q = np.zeros(robot.nq)

# Forward kinematics
T = robot.fkine(q)  # 4x4 homogeneous transform

# World-frame Jacobian
J = robot.jacob0(q)  # 6 x nq, [linear; angular]

# Joint limits (2 x nq)
qlim = robot.qlim  # row 0: lower, row 1: upper

# Inverse kinematics
solver = IKSolver(robot)
solver.solve(T, q0=q)
print(solver.q, solver.success)
```

### Zero-allocation variants for hot loops

```python
T_buf = np.empty((4, 4))
J_buf = np.empty((6, robot.nq))

robot.fkine_into(q, T_buf)
robot.jacob0_into(q, J_buf)
```

## Build from source

Requires conda for the Pinocchio dependency:

```bash
conda env create -f environment.yml
conda activate pinokin
pip install -e ".[dev]" --no-build-isolation
pytest tests/ -v
```

## Acknowledgments

- **IK algorithms** (Gauss-Newton, Newton-Raphson, Levenberg-Marquardt with Chan/Wampler/Sugihara damping, angle_axis error) ported from [robotics-toolbox-python](https://github.com/petercorke/robotics-toolbox-python) by Peter Corke et al., MIT license.
- **FK, Jacobians, and URDF parsing** powered by [Pinocchio](https://github.com/stack-of-tasks/pinocchio), BSD-2 license.
- **Python bindings** via [nanobind](https://github.com/wjakob/nanobind) by Wenzel Jakob, BSD-3 license.
