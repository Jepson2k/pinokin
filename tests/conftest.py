from pathlib import Path

import pytest

from pinokin import CollisionChecker, IKSolver, Robot

URDF_PATH = str(Path(__file__).parent / "parol6.urdf")
COLLISION_URDF_PATH = str(Path(__file__).parent / "collision_two_link.urdf")


@pytest.fixture(scope="session")
def robot():
    return Robot(URDF_PATH)


@pytest.fixture
def solver(robot):
    return IKSolver(robot)


@pytest.fixture(scope="session")
def collision_robot():
    """Self-contained primitives-only robot for collision tests."""
    return Robot(COLLISION_URDF_PATH)


@pytest.fixture(scope="session")
def checker(collision_robot):
    """CollisionChecker bound to the primitives URDF."""
    return CollisionChecker(collision_robot, COLLISION_URDF_PATH)
