"""Grid sweep generator — rasters through joint space with smooth ramps."""
from __future__ import annotations

import itertools

import numpy as np

from .base import InputGenerator


class SweepGenerator(InputGenerator):
    """Raster sweep through a grid of (insertion, rotation, cable) waypoints.

    Generates smooth linear ramps between grid points so SOFA's solver stays
    stable.  At each waypoint the catheter dwells for ``dwell_time`` seconds
    to let transients settle before the collector records a quasi-static
    snapshot.

    Parameters
    ----------
    joint_lower_limits, joint_upper_limits:
        Length-3 arrays [insertion, rotation, cable] bounds.
    dt:
        Simulation timestep (seconds).
    n_insertion, n_rotation, n_cable:
        Number of grid points along each axis.
    ramp_speed:
        Fraction of joint range traversed per second during ramp segments.
        e.g. 0.5 means the full range is swept in 2 seconds.
    dwell_time:
        Seconds to dwell at each waypoint.
    """

    def __init__(
        self,
        joint_lower_limits: np.ndarray,
        joint_upper_limits: np.ndarray,
        dt: float,
        n_insertion: int = 5,
        n_rotation: int = 7,
        n_cable: int = 10,
        ramp_speed: float = 0.3,
        dwell_time: float = 0.5,
    ) -> None:
        super().__init__(joint_lower_limits, joint_upper_limits, dt)
        self._dwell = dwell_time

        ins_pts = np.linspace(self.joint_lower[0], self.joint_upper[0], n_insertion)
        rot_pts = np.linspace(self.joint_lower[1], self.joint_upper[1], n_rotation)
        cab_pts = np.linspace(self.joint_lower[2], self.joint_upper[2], n_cable)

        waypoints = list(itertools.product(ins_pts, rot_pts, cab_pts))
        self._waypoints = [np.array(w) for w in waypoints]

        joint_range = self.joint_upper - self.joint_lower
        joint_range[joint_range < 1e-12] = 1.0
        self._ramp_rate = ramp_speed * joint_range

        self._segments = self._build_segments()
        self._total_time = sum(s[0] for s in self._segments) if self._segments else 0.0

    def _build_segments(self):
        """Build list of (duration, start_pos, end_pos) segments."""
        if not self._waypoints:
            return []
        segments = []
        prev = self._waypoints[0].copy()
        segments.append((self._dwell, prev, prev.copy()))
        for wp in self._waypoints[1:]:
            delta = np.abs(wp - prev)
            safe_rate = np.where(self._ramp_rate > 1e-12, self._ramp_rate, 1.0)
            ramp_time = max(np.max(delta / safe_rate), self.dt)
            segments.append((ramp_time, prev.copy(), wp.copy()))
            segments.append((self._dwell, wp.copy(), wp.copy()))
            prev = wp.copy()
        return segments

    def step(self, t: float) -> np.ndarray:
        elapsed = 0.0
        for duration, start, end in self._segments:
            if t < elapsed + duration:
                alpha = (t - elapsed) / duration if duration > 0 else 1.0
                alpha = np.clip(alpha, 0.0, 1.0)
                return start + alpha * (end - start)
            elapsed += duration
        return self._waypoints[-1] if self._waypoints else self.joint_lower.copy()

    def is_done(self, t: float) -> bool:
        return t >= self._total_time
