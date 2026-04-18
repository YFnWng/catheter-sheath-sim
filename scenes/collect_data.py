"""Data collection scene — drives the catheter with scripted trajectories
and records per-timestep state to HDF5.

Supports multi-scene batch collection: the YAML ``scenes`` list defines
multiple environment configurations.  Each scene is run as a separate
SOFA session with its own trajectory and output file.

Two execution modes:

1. **GUI** (runSofa) — runs the first scene with visual debugging + plotter:
       runSofa -g qt simulation/scenes/collect_data.py

2. **True headless** (Sofa.Simulation.animate) — runs ALL scenes sequentially:
       python simulation/scenes/collect_data.py

Environment variables
---------------------
COLLECT_GENERATOR : str
    Generator type: ``sweep`` (default) or ``sinusoidal``.
COLLECT_OUTPUT : str
    Output HDF5 path (default: auto-generated per scene).
COLLECT_WARMUP : int
    Warmup steps before recording (default: 50).
COLLECT_MAX_STEPS : int
    Maximum simulation steps per scene (default: unlimited).
COLLECT_SCENE_IDX : int
    Scene index to run in GUI mode (default: 0).
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

from utils.sofa_env import ensure_sofa_paths
ensure_sofa_paths()

import Sofa
import Sofa.Core
import Sofa.Simulation

from utils.scene import add_required_plugins, add_scene_utilities
from utils.message_handler import SofaMessageHandler
from objects.fixed_rigid_body import add_environment
from robots.catheter import CatheterRobot

from data_collection.generators import SweepGenerator, SinusoidalGenerator
from data_collection.collector import DataCollectorController

from utils.sofa_reader import SofaReader
from utils.plotter import DiagnosticPlotter
from controllers.plot_controller import PlotController

_CONFIG_PATH = os.path.join(_SIM_DIR, "configs", "catheter_ablation.yaml")

_GENERATORS = {
    "sweep": SweepGenerator,
    "sinusoidal": SinusoidalGenerator,
}


def createScene(root: Sofa.Core.Node, headless: bool = False,
                scene_objects=None) -> Sofa.Core.Node:
    """Build one data-collection scene.

    Parameters
    ----------
    root : Sofa.Core.Node
    headless : bool
        If True, skip the diagnostic plotter.
    scene_objects : list of str or None
        Environment model names.  If None, uses the first entry in
        ``cfg["scenes"]`` or falls back to ``["PipelineModel"]``.
    """
    gen_name = os.environ.get("COLLECT_GENERATOR", "sweep")
    output_path = os.environ.get("COLLECT_OUTPUT", "")
    warmup_steps = int(os.environ.get("COLLECT_WARMUP", "0"))

    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    if scene_objects is None:
        scenes = cfg.get("scenes", [["PipelineModel"]])
        scene_idx = int(os.environ.get("COLLECT_SCENE_IDX", "0"))
        scene_objects = scenes[min(scene_idx, len(scenes) - 1)]

    scene_tag = "_".join(scene_objects) if isinstance(scene_objects[0], str) else "custom"
    if not output_path:
        data_dir = os.path.join(_SIM_DIR, "data_collection", "data")
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(data_dir, f"{gen_name}_{scene_tag}_{ts}.h5")

    root.gravity = [0.0, 0.0, 0.0]
    root.dt = 0.01

    add_required_plugins(root)
    add_scene_utilities(root)

    cable_mode = cfg.get("actuation", {}).get("cable_mode", "force")

    add_environment(root, scene_objects)
    robot = CatheterRobot(root, config_path=_CONFIG_PATH, cable_mode=cable_mode)

    reader = SofaReader(
        prefab=robot._prefab,
        base_mo=robot.base_mo,
        cable_constraints=robot.cable_constraints,
        prefab_rotation_offset=robot.prefab_rotation_offset,
        cable_mode=cable_mode,
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
        "scene_objects": scene_objects,
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

    if not headless:
        rod_cfg = cfg.get("rod", {})
        n_nodes = int(rod_cfg.get("n_frames", 33))
        n_sections = int(rod_cfg.get("n_sections", 32))
        n_cables = len(cfg.get("actuation", {}).get("cable_locations", [[0, 0]]))

        plotter = DiagnosticPlotter(
            n_nodes=n_nodes,
            n_sections=n_sections,
            rod_length=float(rod_cfg.get("length", 0.16)),
            panels=("base_translation", "base_rotation",
                    "tendon_force", "contact_force"),
            window_seconds=10.0,
            dt=dt,
            n_cables=n_cables,
            title=f"Data Collection — {scene_tag}",
            size=(1300, 600),
        )
        root.addObject(
            PlotController(
                name="PlotController",
                plotter=plotter,
                reader=reader,
                n_nodes=n_nodes,
                base_home_position=robot.base_position,
            )
        )

    return root


# ---------------------------------------------------------------------------
# True headless entry point — runs ALL scenes sequentially
# ---------------------------------------------------------------------------

def _run_one_scene(scene_objects, scene_idx, total_scenes):
    """Run one scene headless and return (n_steps, wall_time)."""
    max_steps = int(os.environ.get("COLLECT_MAX_STEPS", "0"))
    label = ", ".join(scene_objects) if isinstance(scene_objects[0], str) else f"scene_{scene_idx}"

    root = Sofa.Core.Node("root")
    createScene(root, headless=True, scene_objects=scene_objects)
    Sofa.Simulation.init(root)

    dt = root.dt.value
    controller = root.getObject("DataCollectorController")
    gen = controller._generator
    total_sim_time = getattr(gen, "_total_time", None) or getattr(gen, "_duration", None)
    time_str = f", total_sim_time={total_sim_time:.1f}s" if total_sim_time else ""

    print(f"\n[collect_data] Scene {scene_idx + 1}/{total_scenes}: [{label}]{time_str}")

    step = 0
    t0 = time.time()

    while not controller._done:
        Sofa.Simulation.animate(root, dt)
        step += 1
        if step % 500 == 0:
            elapsed = time.time() - t0
            print(f"  step {step}, sim_time={step * dt:.2f}s, "
                  f"wall_time={elapsed:.1f}s", flush=True)
        if max_steps > 0 and step >= max_steps:
            print(f"  Reached max_steps={max_steps}, forcing save.")
            controller._finish()
            break

    elapsed = time.time() - t0
    print(f"  Done: {step} steps in {elapsed:.1f}s ({step / elapsed:.0f} steps/s)")

    Sofa.Simulation.unload(root)
    return step, elapsed


def run_headless():
    """Run all scenes from the YAML config sequentially."""
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    scenes = cfg.get("scenes", [["PipelineModel"]])
    print(f"[collect_data] {len(scenes)} scene(s) to collect")

    msg_handler = SofaMessageHandler(print_info=False)
    total_steps = 0
    total_time = 0.0

    with msg_handler:
        for i, scene_objects in enumerate(scenes):
            steps, elapsed = _run_one_scene(scene_objects, i, len(scenes))
            total_steps += steps
            total_time += elapsed

    print(f"\n[collect_data] All done: {total_steps} total steps "
          f"in {total_time:.1f}s across {len(scenes)} scene(s)")
    print(f"[collect_data] {msg_handler.summary()}")


if __name__ == "__main__":
    run_headless()
