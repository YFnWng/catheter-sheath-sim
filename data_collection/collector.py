"""DataCollectorController — SOFA Controller that drives the catheter
with an InputGenerator and records per-timestep state via SofaReader.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

import Sofa
import Sofa.Core

from .generators.base import InputGenerator
from .schema import TrajectoryRecord, write_hdf5


class DataCollectorController(Sofa.Core.Controller):
    """Automated data collection controller.

    Replaces ``CatheterKeyboardController`` for scripted trajectory execution.
    Reads per-timestep state via a ``SofaReader`` and buffers it for HDF5
    export.

    Parameters (passed as kwargs)
    ----------
    generator : InputGenerator
        Produces joint commands at each timestep.
    reader : object
        A ``SofaReader`` instance (from ``state_estimation.sofa.bridge.reader``).
    base_mechanical_object : SOFA MechanicalObject
        The Rigid3d base DOF node (``robot.base_mo``).
    cable_constraint : SOFA object or None
        First cable constraint for writing cable commands.
    direction, base_position, base_orientation, prefab_rotation_offset :
        Same semantics as ``CatheterKeyboardController``.
    output_path : str
        Where to write the HDF5 file.
    warmup_steps : int
        Steps to skip before recording (let transients settle).
    metadata : dict or None
        Extra metadata to store in HDF5.
    """

    _SOFA_CABLE_SCALE = 0.01

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._generator: InputGenerator = kwargs.pop("generator")
        self._reader = kwargs.pop("reader")
        self._base_mo = kwargs.pop("base_mechanical_object")
        self._cable_constraint = kwargs.pop("cable_constraint", None)
        self._output_path: str = kwargs.pop("output_path", "trajectory.h5")
        self._warmup_steps: int = kwargs.pop("warmup_steps", 50)
        self._metadata: dict = kwargs.pop("metadata", {})

        self._base_home_position = np.asarray(
            kwargs.pop("base_position", np.zeros(3)), dtype=float,
        )
        base_orient_quat = np.asarray(
            kwargs.pop("base_orientation", [0.0, 0.0, 0.0, 1.0]), dtype=float,
        )
        self._base_home_orientation = R.from_quat(base_orient_quat)
        prefab_rot_quat = np.asarray(
            kwargs.pop("prefab_rotation_offset", [0.0, 0.0, 0.0, 1.0]), dtype=float,
        )
        self._prefab_rot_offset = R.from_quat(prefab_rot_quat)
        direction = np.asarray(kwargs.pop("direction", [0.0, 0.0, 1.0]), dtype=float)
        direction = direction / np.linalg.norm(direction)
        self._direction = direction @ self._base_home_orientation.as_matrix().T

        self._cable_data = None
        if self._cable_constraint is not None:
            self._cable_data = self._cable_constraint.findData("value")

        self._record = TrajectoryRecord()
        self._step_count = 0
        self._done = False
        self._t_start: Optional[float] = None

    def onAnimateBeginEvent(self, _event) -> None:
        if self._done:
            return
        if self._base_mo is None or len(self._base_mo.position.value) == 0:
            return

        self._step_count += 1

        # During warmup, hold at initial position
        if self._step_count <= self._warmup_steps:
            return

        dt = float(self._base_mo.getContext().dt.value)
        t_sim = float(self._base_mo.getContext().time.value)
        if self._t_start is None:
            self._t_start = t_sim

        t = t_sim - self._t_start

        if self._generator.is_done(t):
            self._finish()
            return

        joint_cmd = self._generator.step(t)
        joint_cmd = np.clip(joint_cmd, self._generator.joint_lower,
                            self._generator.joint_upper)
        self._apply_joint_commands(joint_cmd)

    def onAnimateEndEvent(self, _event) -> None:
        if self._done or self._step_count <= self._warmup_steps:
            return

        t_sim = float(self._base_mo.getContext().time.value)
        sofa_gt = self._reader.read()
        t = t_sim - self._t_start if self._t_start is not None else 0.0

        if self._generator.is_done(t):
            return

        joint_cmd = self._generator.step(t)
        joint_cmd = np.clip(joint_cmd, self._generator.joint_lower,
                            self._generator.joint_upper)

        self._record.append(
            t=t_sim,
            frame_poses=sofa_gt.frame_poses,
            strain_coords=sofa_gt.strain_coords,
            cable_tensions=sofa_gt.cable_tensions,
            joint_commands=joint_cmd,
            contact_force_body=sofa_gt.contact_force_body,
        )

    def _apply_joint_commands(self, joint_cmd: np.ndarray) -> None:
        """Apply [insertion, rotation, cable] to SOFA objects."""
        insertion = joint_cmd[0]
        rotation_deg = joint_cmd[1]
        cable_val = joint_cmd[2]

        translation = self._direction * insertion
        rotation = R.from_rotvec(rotation_deg * self._direction, degrees=True)
        base_orientation = (
            rotation * self._base_home_orientation * self._prefab_rot_offset
        ).as_quat()

        with self._base_mo.position.writeable() as pos:
            pos[0][0:3] = (self._base_home_position + translation).tolist()
            pos[0][3:7] = base_orientation.tolist()
        if (hasattr(self._base_mo, "rest_position")
                and len(self._base_mo.rest_position.value) > 0):
            with self._base_mo.rest_position.writeable() as rest:
                rest[0][0:3] = (self._base_home_position + translation).tolist()
                rest[0][3:7] = base_orientation.tolist()

        if self._cable_data is not None:
            self._cable_data.value = [cable_val * self._SOFA_CABLE_SCALE]

    def _finish(self) -> None:
        self._done = True
        n = len(self._record.timestamps)
        print(f"[DataCollector] Trajectory complete: {n} timesteps recorded.")
        os.makedirs(os.path.dirname(os.path.abspath(self._output_path)), exist_ok=True)
        meta = {
            "generator": self._generator.name,
            "n_timesteps": n,
            **self._metadata,
        }
        write_hdf5(self._output_path, self._record, metadata=meta)
        print(f"[DataCollector] Saved to {self._output_path}")
