import os
import sys

# Make simulation root importable regardless of how runSofa sets sys.path
_SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)

import Sofa
import Sofa.Core

from utils.scene import add_required_plugins, add_scene_utilities
from objects.heart import HeartModel
from robots.catheter import CatheterRobot
from controllers.keyboard_controller import CatheterKeyboardController


def createScene(root: Sofa.Core.Node) -> Sofa.Core.Node:
    root.gravity = [0.0, -9.81, 0.0]
    root.dt = 0.01

    add_required_plugins(root)
    add_scene_utilities(root)

    heart = HeartModel(root)
    robot = CatheterRobot(root)

    root.addObject(
        "ContactListener",
        name="contactStats",
        collisionModel1=heart.triangle_collision_model_path,
        collisionModel2=robot.point_collision_model_path,
    )

    root.addObject(
        CatheterKeyboardController(
            name="CatheterKeyboardController",
            base_mechanical_object=robot.base_mo,
            direction=robot.insertion_direction,
            joint_rate=robot.joint_rate,
            joint_upper_limits=robot.joint_upper_limits,
            joint_lower_limits=robot.joint_lower_limits,
            base_position=robot.base_position,
            base_orientation=robot.base_orientation,
            cable_constraint=robot.cable_constraint,
        )
    )

    return root
