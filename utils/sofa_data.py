"""SofaGroundTruth dataclass — raw simulation state extracted from SOFA."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SofaGroundTruth:
    """Raw simulation state extracted from SOFA.

    Attributes
    ----------
    frame_poses : (n_frames, 7)
        Rigid3d [x, y, z, qx, qy, qz, qw] per Cosserat frame.
    strain_coords : (n_sections, 3)
        Curvature/torsion per section.
    base_pose : (7,)
        Base frame pose [x, y, z, qx, qy, qz, qw].
    cable_disp : float
        Cable displacement — displacement-controlled mode.
    cable_tensions : (n_cables,) or None
        Per-cable tensions (N) — force-controlled mode.
    contact_force_body : (n_nodes, 3)
        Per-node contact force in the rod body frame (N).
    """

    frame_poses: np.ndarray
    strain_coords: np.ndarray
    base_pose: np.ndarray
    cable_disp: float = 0.0
    cable_tensions: Optional[np.ndarray] = None
    contact_force_body: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3))
    )
