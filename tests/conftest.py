from pathlib import Path

import pytest

from pinokin import Robot, IKSolver

URDF_PATH = str(Path(__file__).parent / "parol6.urdf")


@pytest.fixture(scope="session")
def robot():
    return Robot(URDF_PATH)


@pytest.fixture
def solver(robot):
    return IKSolver(robot)
