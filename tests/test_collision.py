"""Smoke tests for pinokin.CollisionChecker.

Uses the synthetic primitives URDF (tests/collision_two_link.urdf) so the
suite doesn't depend on external mesh files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pinokin import CollisionChecker

COLLISION_URDF_PATH = str(Path(__file__).parent / "collision_two_link.urdf")


def test_load_geom_from_urdf(checker):
    # 3 collision boxes (base_link, L1, L2). Some pairs disabled as adjacent.
    assert checker.num_geometry_objects == 3
    # base_link <-> L2 is non-adjacent (separated by L1), so at least 1 pair.
    assert checker.num_collision_pairs >= 1


def test_home_is_clear(collision_robot, checker):
    q = np.zeros(collision_robot.nq)
    assert checker.in_collision(q) is False


def test_folded_self_collision(collision_robot, checker):
    # q = [0, pi] folds L2 back so it intersects L1/base region.
    q = np.array([0.0, np.pi])
    assert checker.in_collision(q) is True
    pairs = checker.colliding_pairs(q)
    assert len(pairs) >= 1


def test_segment_check_clear_to_colliding(collision_robot, checker):
    q0 = np.zeros(collision_robot.nq)
    q1 = np.array([0.0, np.pi])
    assert checker.check_segment(q0, q1, n_steps=10) is True


def test_segment_check_clear_to_clear(collision_robot, checker):
    q0 = np.zeros(collision_robot.nq)
    q1 = np.array([0.5, 0.5])
    assert checker.check_segment(q0, q1, n_steps=10) is False


def test_check_path_returns_first_colliding(collision_robot, checker):
    q_path = np.array([
        [0.0, 0.0],
        [0.0, 0.5],
        [0.0, 1.0],
        [0.0, np.pi],
        [0.0, np.pi],
    ])
    idx = checker.check_path(q_path)
    assert idx == 3


def test_check_path_all_clear(collision_robot, checker):
    q_path = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.2],
    ])
    assert checker.check_path(q_path) == -1


def test_min_distance_positive_at_home(collision_robot, checker):
    q = np.zeros(collision_robot.nq)
    d = checker.min_distance(q)
    assert d > 0.0


@pytest.fixture
def fresh_checker(collision_robot):
    """A fresh CollisionChecker per test so mutations don't leak."""
    return CollisionChecker(collision_robot, COLLISION_URDF_PATH)


def test_add_obstacle_box_clear(collision_robot, fresh_checker):
    pose = np.eye(4)
    pose[2, 3] = -2.0
    fresh_checker.add_obstacle_box("table", np.array([0.5, 0.5, 0.02]), pose)
    assert fresh_checker.in_collision(np.zeros(collision_robot.nq)) is False


def test_add_obstacle_box_collides(collision_robot, fresh_checker):
    pose = np.eye(4)
    pose[2, 3] = 0.25
    fresh_checker.add_obstacle_box("wall", np.array([0.5, 0.5, 0.2]), pose)
    assert fresh_checker.in_collision(np.zeros(collision_robot.nq)) is True


def test_remove_obstacle_clears(collision_robot, fresh_checker):
    pose = np.eye(4)
    pose[2, 3] = 0.25
    fresh_checker.add_obstacle_box("wall", np.array([0.5, 0.5, 0.2]), pose)
    assert fresh_checker.in_collision(np.zeros(collision_robot.nq)) is True
    fresh_checker.remove_geometry_by_name("wall")
    assert fresh_checker.in_collision(np.zeros(collision_robot.nq)) is False


def test_move_obstacle(collision_robot, fresh_checker):
    pose_far = np.eye(4)
    pose_far[2, 3] = -2.0
    fresh_checker.add_obstacle_box("wall", np.array([0.5, 0.5, 0.2]), pose_far)
    assert fresh_checker.in_collision(np.zeros(collision_robot.nq)) is False
    pose_near = np.eye(4)
    pose_near[2, 3] = 0.25
    fresh_checker.set_geometry_pose_by_name("wall", pose_near)
    assert fresh_checker.in_collision(np.zeros(collision_robot.nq)) is True


def test_obstacle_sphere_collides(collision_robot, fresh_checker):
    pose = np.eye(4)
    pose[2, 3] = 0.20
    fresh_checker.add_obstacle_sphere("ball", 0.30, pose)
    assert fresh_checker.in_collision(np.zeros(collision_robot.nq)) is True


def test_obstacle_cylinder_clear_far_away(collision_robot, fresh_checker):
    pose = np.eye(4)
    pose[0, 3] = 5.0
    fresh_checker.add_obstacle_cylinder("post", 0.05, 1.0, pose)
    assert fresh_checker.in_collision(np.zeros(collision_robot.nq)) is False


def test_has_geometry(fresh_checker):
    assert fresh_checker.has_geometry("nope") is False
    fresh_checker.add_obstacle_box("thing", np.array([0.1, 0.1, 0.1]), np.eye(4))
    assert fresh_checker.has_geometry("thing") is True
    fresh_checker.remove_geometry_by_name("thing")
    assert fresh_checker.has_geometry("thing") is False


def test_attach_box_to_wrist_clear_at_home(collision_robot, fresh_checker):
    placement = np.eye(4)
    placement[2, 3] = 0.25
    fresh_checker.attach_box_to_frame(
        "gripper", np.array([0.03, 0.03, 0.05]), "L2", placement
    )
    assert fresh_checker.in_collision(np.zeros(collision_robot.nq)) is False


def test_attached_payload_collides_when_folded(collision_robot, fresh_checker):
    placement = np.eye(4)
    placement[2, 3] = 0.30
    fresh_checker.attach_box_to_frame(
        "payload", np.array([0.05, 0.05, 0.05]), "L2", placement
    )
    q = np.array([0.0, np.pi])
    assert fresh_checker.in_collision(q) is True


def test_release_payload_to_world(collision_robot, fresh_checker):
    placement = np.eye(4)
    placement[2, 3] = 0.30
    fresh_checker.attach_box_to_frame(
        "payload", np.array([0.05, 0.05, 0.05]), "L2", placement
    )
    q_home = np.zeros(collision_robot.nq)
    fresh_checker.update_placements(q_home)
    world_pose = fresh_checker.geometry_world_pose("payload")
    fresh_checker.reparent_geometry_by_name("payload", "universe", world_pose)
    assert fresh_checker.in_collision(q_home) is False
