"""HDF5 trajectory schema — dataclass + I/O utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import json
import numpy as np


@dataclass
class TrajectoryRecord:
    """In-memory buffer for one trajectory's worth of timestep data.

    All arrays are lists-of-arrays during collection and converted to
    contiguous numpy arrays by :meth:`finalise` before HDF5 writing.

    Fields
    ------
    timestamps : simulation time (s) — recorded at begin of step
    frame_poses : (n_frames, 7) Rigid3d per frame
    strain_coords : (n_sections, 3) curvature per section
    joint_commands : (n_joints,) control input [insertion, rotation, cable, ...]
    contact_force_body : (n_frames, 3) per-node contact force in body frame
    tip_force : (3,) random external force applied to tip [Fx, Fy, Fz]
    frame_velocity : (n_frames, 6) Rigid3d velocity per frame
    strain_velocity : (n_sections, 3) strain rate per section
    """
    timestamps: List[float] = field(default_factory=list)
    frame_poses: List[np.ndarray] = field(default_factory=list)
    strain_coords: List[np.ndarray] = field(default_factory=list)
    joint_commands: List[np.ndarray] = field(default_factory=list)
    contact_force_body: List[np.ndarray] = field(default_factory=list)
    tip_force: List[np.ndarray] = field(default_factory=list)
    frame_velocity: List[np.ndarray] = field(default_factory=list)
    strain_velocity: List[np.ndarray] = field(default_factory=list)

    def append(
        self,
        t: float,
        frame_poses: np.ndarray,
        strain_coords: np.ndarray,
        joint_commands: np.ndarray,
        contact_force_body: np.ndarray,
        tip_force: np.ndarray = None,
        frame_velocity: np.ndarray = None,
        strain_velocity: np.ndarray = None,
    ) -> None:
        self.timestamps.append(t)
        self.frame_poses.append(frame_poses.copy())
        self.strain_coords.append(strain_coords.copy())
        self.joint_commands.append(joint_commands.copy())
        self.contact_force_body.append(contact_force_body.copy())
        self.tip_force.append(tip_force.copy() if tip_force is not None
                              else np.zeros(3))
        if frame_velocity is not None:
            self.frame_velocity.append(frame_velocity.copy())
        if strain_velocity is not None:
            self.strain_velocity.append(strain_velocity.copy())

    def finalise(self) -> Dict[str, np.ndarray]:
        """Stack lists into contiguous arrays for HDF5 writing."""
        arrays = {
            "timestamps": np.array(self.timestamps, dtype=np.float64),
            "frame_poses": np.stack(self.frame_poses).astype(np.float64),
            "strain_coords": np.stack(self.strain_coords).astype(np.float64),
            "joint_commands": np.stack(self.joint_commands).astype(np.float64),
            "contact_force_body": np.stack(self.contact_force_body).astype(np.float64),
            "tip_force": np.stack(self.tip_force).astype(np.float64),
        }
        if self.frame_velocity:
            arrays["frame_velocity"] = np.stack(self.frame_velocity).astype(np.float64)
        if self.strain_velocity:
            arrays["strain_velocity"] = np.stack(self.strain_velocity).astype(np.float64)
        return arrays


def write_hdf5(
    path: str,
    record: TrajectoryRecord,
    metadata: Optional[Dict] = None,
    extra: Optional[Dict] = None,
) -> None:
    """Write a finalised trajectory record to an HDF5 file.

    Parameters
    ----------
    extra : dict of str -> (list or np.ndarray), optional
        Additional datasets to write (e.g., control-specific data).
        Lists are converted to numpy arrays automatically.
        Empty lists are skipped.
    """
    import h5py

    arrays = record.finalise()
    with h5py.File(path, "w") as f:
        for name, arr in arrays.items():
            f.create_dataset(name, data=arr, compression="gzip", compression_opts=4)
        if extra:
            for name, vals in extra.items():
                if vals is None or (hasattr(vals, '__len__') and len(vals) == 0):
                    continue
                arr = np.asarray(vals, dtype=np.float64)
                f.create_dataset(name, data=arr, compression="gzip",
                                 compression_opts=4)
        if metadata:
            f.attrs["metadata"] = json.dumps(metadata)


def read_hdf5(path: str):
    """Read an HDF5 trajectory file. Returns (dict of arrays, metadata dict)."""
    import h5py

    with h5py.File(path, "r") as f:
        arrays = {name: np.array(ds) for name, ds in f.items()}
        meta_str = f.attrs.get("metadata", "{}")
        metadata = json.loads(meta_str)
    return arrays, metadata
