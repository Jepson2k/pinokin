# pinokin

FK, Jacobians, and IK for URDF robots — Pinocchio C++ backend with nanobind Python bindings.

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

# Inverse kinematics
solver = IKSolver(robot)
solver.solve(T, q0=q)
print(solver.q, solver.success)
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

- **IK algorithms** (Gauss-Newton, Newton-Raphson, Levenberg-Marquardt with Chan/Wampler/Sugihara damping, angle_axis error, smart wrapping) ported from [robotics-toolbox-python](https://github.com/petercorke/robotics-toolbox-python) by Peter Corke et al., MIT license.
- **FK, Jacobians, and URDF parsing** powered by [Pinocchio](https://github.com/stack-of-tasks/pinocchio), BSD-2 license.
- **Python bindings** via [nanobind](https://github.com/wjakob/nanobind) by Wenzel Jakob, BSD-3 license.
