"""Visualize a task environment and initial robot configuration.

Loads a single scene from generated_scenes.yaml and renders the robot
at its initial configuration.  Same setup as collect_data.py but without
data collection — purely for visual inspection.

Usage::

    runSofa simulation/scenes/view_all_tasks.py

    # Specify task index via env var (default: 1):
    TASK_IDX=3 runSofa simulation/scenes/view_all_tasks.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import yaml

# runSofa may execute the file as a string — __file__ may not be set.
def _find_sim_dir():
    try:
        return os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".."))
    except NameError:
        pass
    for arg in sys.argv:
        if "view_all_tasks.py" in arg and os.path.isfile(arg):
            return os.path.normpath(os.path.join(
                os.path.dirname(os.path.abspath(arg)), ".."))
    if os.path.isdir(os.path.join(os.getcwd(), "..", "utils")):
        return os.path.normpath(os.path.join(os.getcwd(), ".."))
    for candidate in [os.path.expanduser("~/Yifan/simulation"),
                      "/home/chen-lab/Yifan/simulation"]:
        if os.path.isdir(os.path.join(candidate, "utils")):
            return candidate
    raise RuntimeError("Cannot locate simulation/ directory")


_SIM_DIR = _find_sim_dir()
_WORKSPACE = os.path.normpath(os.path.join(_SIM_DIR, ".."))

if _SIM_DIR in sys.path:
    sys.path.remove(_SIM_DIR)
sys.path.insert(0, _SIM_DIR)
if _WORKSPACE not in sys.path:
    sys.path.insert(1, _WORKSPACE)

from utils.sofa_env import ensure_sofa_paths
ensure_sofa_paths()

import Sofa
import Sofa.Core
import Sofa.Simulation

from utils.scene import add_required_plugins, add_scene_utilities
from objects.fixed_rigid_body import add_environment
from robots.catheter import CatheterRobot

_ROBOT_CONFIG_PATH = os.path.join(
    _WORKSPACE, "simulation", "configs", "catheter_ablation.yaml",
)
_DEFAULT_SCENES_YAML = os.path.join(
    _WORKSPACE, "simulation", "configs", "generated_scenes.yaml",
)


class _InitApplier(Sofa.Core.Controller):
    """Apply initial config on the first animation step (after SOFA init)."""

    def __init__(self, name, robot, strain_mo, initial_config, **kwargs):
        super().__init__(name=name, **kwargs)
        self._robot = robot
        self._strain_mo = strain_mo
        self._initial_config = initial_config
        self._applied = False

    def onAnimateBeginEvent(self, event):
        if self._applied:
            return
        self._applied = True

        ic = self._initial_config
        if ic is None:
            return

        bp = ic.get("base_pose")
        if bp is not None:
            bp = np.array(bp, dtype=float)
            with self._robot.base_mo.position.writeable() as pos:
                pos[0][:] = bp
            if (hasattr(self._robot.base_mo, "rest_position")
                    and len(self._robot.base_mo.rest_position.value) > 0):
                with self._robot.base_mo.rest_position.writeable() as rest:
                    rest[0][:] = bp

        sc = ic.get("strain_coords")
        if sc is not None and self._strain_mo is not None:
            sc_arr = [np.array(s, dtype=float) for s in sc]
            with self._strain_mo.position.writeable() as strain_pos:
                n = min(len(sc_arr), len(strain_pos))
                for i in range(n):
                    strain_pos[i][:] = sc_arr[i]
            if hasattr(self._strain_mo, "rest_position"):
                with self._strain_mo.rest_position.writeable() as rest:
                    n = min(len(sc_arr), len(rest))
                    for i in range(n):
                        rest[i][:] = sc_arr[i]

        print(f"[view_all_tasks] Applied initial config")


def createScene(root: Sofa.Core.Node) -> Sofa.Core.Node:
    """Build a scene with one robot at a selected task's init config."""

    # ── Load scenes ──────────────────────────────────────────────────
    scenes_yaml = os.environ.get("SCENES_YAML", _DEFAULT_SCENES_YAML)
    with open(scenes_yaml) as f:
        all_scenes = yaml.safe_load(f).get("scenes", [])
    n_scenes = len(all_scenes)

    task_idx = int(os.environ.get("TASK_IDX", "1"))
    task_idx = max(0, min(task_idx, n_scenes - 1))

    scene_dict = all_scenes[task_idx]
    initial_config = scene_dict.get("initial_config")

    obj_types = [o.get("type", "?") for o in scene_dict.get("objects", [])]
    label = ", ".join(obj_types) if obj_types else "free space"
    print(f"[view_all_tasks] Robot: task {task_idx}/{n_scenes - 1} ({label})")
    print(f"  Objects: all {n_scenes} scenes")

    # ── Scene setup ──────────────────────────────────────────────────
    root.gravity = [0.0, 0.0, 0.0]
    root.dt = 0.01

    add_required_plugins(root)
    add_scene_utilities(root)

    with open(_ROBOT_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cable_mode = cfg.get("actuation", {}).get("cable_mode", "force")

    # ── Obstacles from ALL scenes ───────────────────────────────────
    for i, sd in enumerate(all_scenes):
        objs = sd.get("objects", [])
        if objs:
            add_environment(root, objs)

    # ── Robot ────────────────────────────────────────────────────────
    robot = CatheterRobot(root, config_path=_ROBOT_CONFIG_PATH,
                          cable_mode=cable_mode)
    strain_mo = robot._prefab.cosseratCoordinate.cosseratCoordinateMO

    # ── Apply init config after SOFA init ────────────────────────────
    root.addObject(
        _InitApplier(
            name="InitApplier",
            robot=robot,
            strain_mo=strain_mo,
            initial_config=initial_config,
        )
    )

    return root
