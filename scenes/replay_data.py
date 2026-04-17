"""Replay a recorded HDF5 trajectory in SOFA — no physics, just visualization.

Reads frame poses from a collected trajectory file and animates them
step-by-step in SOFA's 3D viewer.  No solver, no contact, no collision
— just a wire-frame rod (+ optional heart mesh) driven by recorded data.

Usage:
    REPLAY_FILE=data_collection/data/sweep_20260417.h5 \
        runSofa -g qt simulation/scenes/replay_data.py

Environment variables
---------------------
REPLAY_FILE : str (required)
    Path to the HDF5 trajectory file.
REPLAY_SPEED : float
    Playback speed multiplier (default: 1.0).  Set > 1 to fast-forward.
REPLAY_SHOW_HEART : str
    Set to ``1`` to show the heart mesh (default: ``1``).
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

if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

import Sofa
import Sofa.Core

from utils.scene import add_required_plugins
from simulation.data_collection.schema import read_hdf5


class ReplayController(Sofa.Core.Controller):
    """Steps through recorded frame poses, updating the visual MO each frame."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frame_poses = kwargs.pop("frame_poses")   # (N, n_frames, 7)
        self._speed = kwargs.pop("speed", 1.0)
        self._loop = kwargs.pop("loop", False)
        self._shape_mo = kwargs.pop("shape_mo")
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


def createScene(root: Sofa.Core.Node) -> Sofa.Core.Node:
    replay_file = os.environ.get("REPLAY_FILE", "")
    if not replay_file:
        raise ValueError("Set REPLAY_FILE env var to the HDF5 trajectory path.")
    if not os.path.isabs(replay_file):
        replay_file = os.path.join(_WORKSPACE, replay_file)

    speed = float(os.environ.get("REPLAY_SPEED", "1.0"))
    show_heart = os.environ.get("REPLAY_SHOW_HEART", "1") == "1"
    loop = os.environ.get("REPLAY_LOOP", "0") == "1"

    arrays, metadata = read_hdf5(replay_file)
    frame_poses = arrays["frame_poses"]   # (N, n_frames, 7)
    n_steps, n_frames, _ = frame_poses.shape

    print(f"[replay] Loaded {replay_file}: {n_steps} steps, {n_frames} frames")

    root.gravity = [0.0, 0.0, 0.0]
    root.dt = 0.01

    add_required_plugins(root)

    root.addObject("VisualStyle",
                    displayFlags="showVisualModels showBehaviorModels")
    root.addObject("BackgroundSetting", color=[1.0, 1.0, 1.0, 1.0])
    root.addObject("OglSceneFrame", style="Arrows", alignment="TopRight")
    root.addObject("DefaultAnimationLoop")

    # Heart mesh (static, for spatial context)
    if show_heart:
        from objects.fixed_rigid_body import HeartModel
        HeartModel(root)

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

    # Replay controller
    root.addObject(
        ReplayController(
            name="ReplayController",
            frame_poses=frame_poses,
            shape_mo=shape_mo,
            speed=speed,
            loop=loop,
        )
    )

    return root
