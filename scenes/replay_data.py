"""Replay a recorded HDF5 trajectory in SOFA — no physics, just visualization.

Reads frame poses from a collected trajectory file and animates them
step-by-step in SOFA's 3D viewer.  Environment objects are reconstructed
from the trajectory's metadata.  A diagnostic plotter shows joint commands
and contact forces alongside the 3D view.

Usage:
    REPLAY_FILE=sweep_PipelineModel_RingModel_20260417.h5 \
        runSofa -g qt simulation/scenes/replay_data.py

REPLAY_FILE is resolved relative to simulation/data_collection/data/ if
not an absolute path and not found relative to cwd.

Environment variables
---------------------
REPLAY_FILE : str (required)
    Path to the HDF5 trajectory file.
REPLAY_SPEED : float
    Playback speed multiplier (default: 1.0).  Set > 1 to fast-forward.
REPLAY_LOOP : str
    Set to ``1`` to loop the replay (default: ``0``).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.normpath(os.path.join(_THIS_DIR, ".."))
_WORKSPACE = os.path.normpath(os.path.join(_SIM_DIR, ".."))
_DATA_DIR = os.path.join(_SIM_DIR, "data_collection", "data")

if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

import Sofa
import Sofa.Core

from utils.scene import add_required_plugins
from objects.fixed_rigid_body import add_environment
from simulation.data_collection.schema import read_hdf5
from utils.plotter import DiagnosticPlotter


def _resolve_replay_file(raw_path: str) -> str:
    """Resolve REPLAY_FILE: try absolute, then cwd, then data_collection/data/."""
    if os.path.isabs(raw_path) and os.path.isfile(raw_path):
        return raw_path
    # Try relative to cwd
    cwd_path = os.path.abspath(raw_path)
    if os.path.isfile(cwd_path):
        return cwd_path
    # Try relative to data_collection/data/
    data_path = os.path.join(_DATA_DIR, raw_path)
    if os.path.isfile(data_path):
        return data_path
    raise FileNotFoundError(
        f"REPLAY_FILE not found: tried {raw_path}, {cwd_path}, {data_path}"
    )


class ReplayController(Sofa.Core.Controller):
    """Steps through recorded data, updating the visual MO and plotter."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frame_poses = kwargs.pop("frame_poses")       # (N, n_frames, 7)
        self._joint_commands = kwargs.pop("joint_commands")  # (N, n_joints)
        self._contact_forces = kwargs.pop("contact_forces")  # (N, n_frames, 3)
        self._timestamps = kwargs.pop("timestamps")          # (N,)
        self._speed = kwargs.pop("speed", 1.0)
        self._loop = kwargs.pop("loop", False)
        self._shape_mo = kwargs.pop("shape_mo")
        self._plotter = kwargs.pop("plotter", None)
        self._n_steps = len(self._frame_poses)
        self._step = 0
        self._substep_accum = 0.0

    def onAnimateBeginEvent(self, _event):
        self._substep_accum += self._speed
        while self._substep_accum >= 1.0:
            self._substep_accum -= 1.0
            self._step += 1

        idx = self._step
        if idx >= self._n_steps:
            if self._loop:
                self._step = 0
                idx = 0
            else:
                return

        positions = self._frame_poses[idx, :, :3]
        self._shape_mo.position.value = positions.tolist()

        if self._plotter is not None:
            n_nodes = self._frame_poses.shape[1]
            jc = self._joint_commands[idx]
            base_pos = np.array([0.0, 0.0, jc[0]]) if len(jc) > 0 else np.zeros(3)
            base_rot = np.array([0.0, jc[1], 0.0]) if len(jc) > 1 else np.zeros(3)
            cable_tensions = jc[2:] if len(jc) > 2 else np.zeros(1)

            gt_F = np.zeros((n_nodes, 3), dtype=np.float32)
            cf = self._contact_forces[idx]
            n = min(n_nodes, len(cf))
            gt_F[:n] = cf[:n]

            self._plotter.send({
                "t": float(self._timestamps[idx]),
                "base_pos": base_pos,
                "base_rot": base_rot,
                "cable_tensions": cable_tensions,
                "gt_F": gt_F,
            })


def createScene(root: Sofa.Core.Node) -> Sofa.Core.Node:
    replay_file = os.environ.get("REPLAY_FILE", "")
    if not replay_file:
        raise ValueError("Set REPLAY_FILE env var to the HDF5 trajectory path.")
    replay_file = _resolve_replay_file(replay_file)

    speed = float(os.environ.get("REPLAY_SPEED", "1.0"))
    loop = os.environ.get("REPLAY_LOOP", "0") == "1"

    arrays, metadata = read_hdf5(replay_file)
    frame_poses = arrays["frame_poses"]
    joint_commands = arrays["joint_commands"]
    contact_forces = arrays.get("contact_force_body",
                                np.zeros((*frame_poses.shape[:2], 3)))
    timestamps = arrays["timestamps"]
    n_steps, n_frames, _ = frame_poses.shape

    scene_objects = metadata.get("scene_objects", [])
    print(f"[replay] Loaded {replay_file}")
    print(f"  {n_steps} steps, {n_frames} frames")
    obj_names = [e.get("type", e) if isinstance(e, dict) else e for e in scene_objects]
    print(f"  scene_objects: {obj_names}")

    dt = metadata.get("dt", 0.01)
    root.gravity = [0.0, 0.0, 0.0]
    root.dt = dt

    add_required_plugins(root)

    root.addObject("VisualStyle",
                    displayFlags="showVisualModels showBehaviorModels")
    root.addObject("BackgroundSetting", color=[1.0, 1.0, 1.0, 1.0])
    root.addObject("OglSceneFrame", style="Arrows", alignment="TopRight")
    root.addObject("DefaultAnimationLoop")

    # Reconstruct environment from metadata
    if scene_objects:
        add_environment(root, scene_objects)

    # Rod visualization — bare Vec3d MO with edges, no solver
    init_pos = frame_poses[0, :, :3].tolist()
    edges = [[k, k + 1] for k in range(n_frames - 1)]

    rod_node = root.addChild("ReplayRod")
    shape_mo = rod_node.addObject(
        "MechanicalObject", name="RodMO", template="Vec3d",
        position=init_pos,
        showObject=True, showObjectScale=3.0,
        showColor=[1.0, 0.2, 0.2, 1.0],
    )
    rod_node.addObject("EdgeSetTopologyContainer", edges=edges)

    rod_visual = rod_node.addChild("Visual")
    rod_visual.addObject(
        "OglModel", name="RodLine",
        color=[1.0, 0.2, 0.2, 1.0],
        edges=edges,
    )
    rod_visual.addObject(
        "IdentityMapping",
        input="@../RodMO", output="@RodLine",
    )

    # Diagnostic plotter
    n_cables = max(1, joint_commands.shape[1] - 2)
    # Estimate rod_length from first frame positions
    positions_0 = frame_poses[0, :, :3]
    rod_length = float(np.sum(np.linalg.norm(np.diff(positions_0, axis=0), axis=1)))

    plotter = DiagnosticPlotter(
        n_nodes=n_frames,
        n_sections=n_frames - 1,
        rod_length=rod_length,
        panels=("base_translation", "base_rotation",
                "tendon_force", "contact_force"),
        window_seconds=min(10.0, n_steps * dt),
        dt=dt,
        n_cables=n_cables,
        update_every=1,
        title=f"Replay — {os.path.basename(replay_file)}",
        size=(1300, 600),
    )

    root.addObject(
        ReplayController(
            name="ReplayController",
            frame_poses=frame_poses,
            joint_commands=joint_commands,
            contact_forces=contact_forces,
            timestamps=timestamps,
            shape_mo=shape_mo,
            plotter=plotter,
            speed=speed,
            loop=loop,
        )
    )

    return root
