"""Data collection scene — drives the catheter with scripted trajectories
and records per-timestep state to HDF5.

Two execution modes:

1. **GUI** (runSofa) — useful for visual debugging:
       runSofa -g qt simulation/scenes/collect_data.py

2. **True headless** (Sofa.Simulation.animate) — no GUI process at all,
   significantly faster for batch collection:
       python simulation/scenes/collect_data.py

Environment variables
---------------------
COLLECT_GENERATOR : str
    Generator type: ``sweep`` (default) or ``sinusoidal``.
COLLECT_OUTPUT : str
    Output HDF5 path (default: ``data_collection/data/<generator>_<timestamp>.h5``).
COLLECT_WARMUP : int
    Warmup steps before recording (default: 50).
COLLECT_MAX_STEPS : int
    Maximum simulation steps (default: unlimited — runs until generator is done).
"""
from __future__ import annotations

import os
import sys
import time
import yaml

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.normpath(os.path.join(_THIS_DIR, ".."))
_WORKSPACE = os.path.normpath(os.path.join(_SIM_DIR, ".."))

if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

import Sofa
import Sofa.Core
import Sofa.Simulation

from utils.scene import add_required_plugins, add_scene_utilities
from utils.message_handler import SofaMessageHandler
from objects.fixed_rigid_body import HeartModel, HeartInsideModel, TurbineModel, PipelineModel
from robots.catheter import CatheterRobot

from data_collection.generators import SweepGenerator, SinusoidalGenerator
from data_collection.collector import DataCollectorController

from state_estimation.sofa.bridge.reader import SofaReader

_CONFIG_PATH = os.path.join(_SIM_DIR, "configs", "catheter_ablation.yaml")

_GENERATORS = {
    "sweep": SweepGenerator,
    "sinusoidal": SinusoidalGenerator,
}


def createScene(root: Sofa.Core.Node) -> Sofa.Core.Node:
    gen_name = os.environ.get("COLLECT_GENERATOR", "sweep")
    output_path = os.environ.get("COLLECT_OUTPUT", "")
    warmup_steps = int(os.environ.get("COLLECT_WARMUP", "50"))

    if not output_path:
        data_dir = os.path.join(_SIM_DIR, "data_collection", "data")
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(data_dir, f"{gen_name}_{ts}.h5")

    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    root.gravity = [0.0, 0.0, 0.0]
    root.dt = 0.01

    add_required_plugins(root)
    add_scene_utilities(root)

    cable_mode = cfg.get("actuation", {}).get("cable_mode", "force")
    env = PipelineModel(root)
    robot = CatheterRobot(root, config_path=_CONFIG_PATH, cable_mode=cable_mode)

    contact_listener = None
    if robot.point_collision_model_path:
        contact_listener = root.addObject(
            "ContactListener",
            name="contactStats",
            collisionModel1=env.triangle_collision_model_path,
            collisionModel2=robot.point_collision_model_path,
        )

    constraint_solver = root.getObject("ConstraintSolver")

    reader = SofaReader(
        prefab=robot._prefab,
        base_mo=robot.base_mo,
        cable_constraints=robot.cable_constraints,
        prefab_rotation_offset=robot.prefab_rotation_offset,
        cable_mode=cable_mode,
        constraint_solver=constraint_solver,
        contact_listener=contact_listener,
    )

    joint_lower = np.array(robot.joint_lower_limits, dtype=float)
    joint_upper = np.array(robot.joint_upper_limits, dtype=float)
    dt = float(root.dt.value)

    gen_cls = _GENERATORS.get(gen_name)
    if gen_cls is None:
        raise ValueError(f"Unknown generator: {gen_name!r}. "
                         f"Available: {list(_GENERATORS.keys())}")
    generator = gen_cls(joint_lower, joint_upper, dt)

    metadata = {
        "config_path": _CONFIG_PATH,
        "cable_mode": cable_mode,
        "dt": dt,
        "gravity": list(root.gravity.value),
    }

    controller = DataCollectorController(
        name="DataCollectorController",
        generator=generator,
        reader=reader,
        base_mechanical_object=robot.base_mo,
        cable_constraint=robot.cable_constraint,
        direction=robot.insertion_direction,
        base_position=robot.base_position,
        base_orientation=robot.base_orientation,
        prefab_rotation_offset=robot.prefab_rotation_offset,
        output_path=output_path,
        warmup_steps=warmup_steps,
        metadata=metadata,
    )
    root.addObject(controller)

    return root


# ---------------------------------------------------------------------------
# True headless entry point — no runSofa, no GUI
# ---------------------------------------------------------------------------

def run_headless():
    """Build scene, step with Sofa.Simulation.animate until done."""
    max_steps = int(os.environ.get("COLLECT_MAX_STEPS", "0"))

    msg_handler = SofaMessageHandler(print_info=False)

    with msg_handler:
        root = Sofa.Core.Node("root")
        createScene(root)
        Sofa.Simulation.init(root)

        dt = root.dt.value
        step = 0
        t0 = time.time()

        controller = root.getObject("DataCollectorController")

        print(f"[collect_data] Running headless simulation (dt={dt})...")
        while not controller._done:
            Sofa.Simulation.animate(root, dt)
            step += 1
            if step % 500 == 0:
                elapsed = time.time() - t0
                print(f"  step {step}, sim_time={step*dt:.2f}s, "
                      f"wall_time={elapsed:.1f}s", flush=True)
            if max_steps > 0 and step >= max_steps:
                print(f"[collect_data] Reached max_steps={max_steps}, forcing save.")
                controller._finish()
                break

        elapsed = time.time() - t0
        print(f"[collect_data] Done: {step} steps in {elapsed:.1f}s "
              f"({step/elapsed:.0f} steps/s)")
        print(f"[collect_data] {msg_handler.summary()}")

        Sofa.Simulation.unload(root)


if __name__ == "__main__":
    run_headless()
