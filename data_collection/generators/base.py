"""Abstract base class for control-input generators."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class InputGenerator(ABC):
    """Generates time-varying joint commands for data collection.

    Subclasses produce trajectories that sweep the catheter through diverse
    configurations so the proximal-boundary basis dictionary is persistently
    excited.

    Joint convention (matching ``CatheterKeyboardController``):
        0 — insertion position (m)
        1 — axial rotation (deg)
        2 — cable value (displacement in m, or force in N depending on mode)
    """

    def __init__(
        self,
        joint_lower_limits: np.ndarray,
        joint_upper_limits: np.ndarray,
        dt: float,
    ) -> None:
        self.joint_lower = np.asarray(joint_lower_limits, dtype=float)
        self.joint_upper = np.asarray(joint_upper_limits, dtype=float)
        self.dt = dt

    @abstractmethod
    def step(self, t: float) -> np.ndarray:
        """Return joint commands [insertion, rotation, cable] at time *t*."""
        ...

    @abstractmethod
    def is_done(self, t: float) -> bool:
        """Return True when the trajectory is complete."""
        ...

    @property
    def name(self) -> str:
        return type(self).__name__
