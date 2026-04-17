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
    """
    timestamps: List[float] = field(default_factory=list)
    frame_poses: List[np.ndarray] = field(default_factory=list)
    strain_coords: List[np.ndarray] = field(default_factory=list)
    cable_tensions: List[Optional[np.ndarray]] = field(default_factory=list)
    joint_commands: List[np.ndarray] = field(default_factory=list)
    contact_force_body: List[np.ndarray] = field(default_factory=list)

    def append(
        self,
        t: float,
        frame_poses: np.ndarray,
        strain_coords: np.ndarray,
        cable_tensions: Optional[np.ndarray],
        joint_commands: np.ndarray,
        contact_force_body: np.ndarray,
    ) -> None:
        self.timestamps.append(t)
        self.frame_poses.append(frame_poses.copy())
        self.strain_coords.append(strain_coords.copy())
        ct = cable_tensions.copy() if cable_tensions is not None else np.zeros(1)
        self.cable_tensions.append(ct)
        self.joint_commands.append(joint_commands.copy())
        self.contact_force_body.append(contact_force_body.copy())

    def finalise(self) -> Dict[str, np.ndarray]:
        """Stack lists into contiguous arrays for HDF5 writing."""
        return {
            "timestamps": np.array(self.timestamps, dtype=np.float64),
            "frame_poses": np.stack(self.frame_poses).astype(np.float64),
            "strain_coords": np.stack(self.strain_coords).astype(np.float64),
            "cable_tensions": np.stack(self.cable_tensions).astype(np.float64),
            "joint_commands": np.stack(self.joint_commands).astype(np.float64),
            "contact_force_body": np.stack(self.contact_force_body).astype(np.float64),
        }


def write_hdf5(
    path: str,
    record: TrajectoryRecord,
    metadata: Optional[Dict] = None,
) -> None:
    """Write a finalised trajectory record to an HDF5 file."""
    import h5py

    arrays = record.finalise()
    with h5py.File(path, "w") as f:
        for name, arr in arrays.items():
            f.create_dataset(name, data=arr, compression="gzip", compression_opts=4)
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
