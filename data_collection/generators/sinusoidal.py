"""Sinusoidal generator — incommensurate-frequency excitation."""
from __future__ import annotations

import numpy as np

from .base import InputGenerator


class SinusoidalGenerator(InputGenerator):
    """Uncorrelated sinusoidal excitation at incommensurate frequencies.

    Each joint oscillates at a different frequency chosen so the products
    (cross-terms like xi*u, s*u) are persistently excited.  Cable output
    is shifted to be non-negative.

    Parameters
    ----------
    joint_lower_limits, joint_upper_limits:
        Length-3 arrays [insertion, rotation, cable] bounds.
    dt:
        Simulation timestep (seconds).
    frequencies:
        (3,) Hz for [insertion, rotation, cable].  Default uses
        incommensurate ratios (0.10, 0.17, 0.31).
    duration:
        Total trajectory duration in seconds.
    """

    def __init__(
        self,
        joint_lower_limits: np.ndarray,
        joint_upper_limits: np.ndarray,
        dt: float,
        frequencies: tuple = (0.10, 0.17, 0.31),
        duration: float = 60.0,
    ) -> None:
        super().__init__(joint_lower_limits, joint_upper_limits, dt)
        self._freq = np.asarray(frequencies, dtype=float)
        self._duration = duration
        self._mid = 0.5 * (self.joint_lower + self.joint_upper)
        self._amp = 0.5 * (self.joint_upper - self.joint_lower)

    def step(self, t: float) -> np.ndarray:
        phase = 2.0 * np.pi * self._freq * t
        return self.wrap_rotation(self._mid + self._amp * np.sin(phase))

    def is_done(self, t: float) -> bool:
        return t >= self._duration
