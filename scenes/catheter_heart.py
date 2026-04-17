import os
import sys
import yaml

# Make simulation root importable regardless of how runSofa sets sys.path
_SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)

import Sofa
import Sofa.Core

from utils.scene import add_required_plugins, add_scene_utilities
from objects.fixed_rigid_body import HeartModel, PipelineModel
from robots.catheter import CatheterRobot
from controllers.keyboard_controller import CatheterKeyboardController

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.normpath(os.path.join(_THIS_DIR, ".."))
_WORKSPACE = os.path.normpath(os.path.join(_SIM_DIR, ".."))

if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

_CONFIG_PATH = os.path.join(_SIM_DIR, "configs", "catheter_ablation.yaml")

def createScene(root: Sofa.Core.Node) -> Sofa.Core.Node:
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    root.gravity = [0.0, 0.0, 0.0]
    root.dt = 0.01

    add_required_plugins(root)
    add_scene_utilities(root)

    cable_mode = cfg.get("actuation", {}).get("cable_mode", "force")
    env = PipelineModel(root)
    robot = CatheterRobot(root, config_path=_CONFIG_PATH, cable_mode=cable_mode)

    root.addObject(
        "ContactListener",
        name="contactStats",
        collisionModel1=env.triangle_collision_model_path,
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
            prefab_rotation_offset=robot.prefab_rotation_offset,
            cable_constraint=robot.cable_constraint,
        )
    )

    return root
