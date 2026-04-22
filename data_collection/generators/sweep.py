"""Grid sweep generator — rasters through joint space with smooth ramps."""
from __future__ import annotations

import itertools

import numpy as np

from .base import InputGenerator


class SweepGenerator(InputGenerator):
    """Raster sweep through a grid of (insertion, rotation, cable) waypoints.

    Generates smooth linear ramps between grid points so SOFA's solver stays
    stable.  Rotation ramps take the shortest angular path and the output
    is wrapped to [-180, 180] degrees.

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
        dwell_time: float = 0.0,
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
        """Build list of (duration, start_pos, delta) segments.

        Uses shortest-path rotation delta so the ramp doesn't travel
        the long way around.
        """
        if not self._waypoints:
            return []
        segments = []
        # Start from zero actuations (matching init config), ramp to first waypoint
        prev = np.zeros_like(self._waypoints[0])
        first_delta = self.shortest_rotation_delta(prev, self._waypoints[0])
        abs_first = np.abs(first_delta)
        safe_rate = np.where(self._ramp_rate > 1e-12, self._ramp_rate, 1.0)
        if np.max(abs_first) > 1e-8:
            ramp_to_first = max(np.max(abs_first / safe_rate), self.dt)
            segments.append((ramp_to_first, prev.copy(), first_delta))
            prev = prev + first_delta
        segments.append((self._dwell, prev.copy(), np.zeros_like(prev)))
        for wp in self._waypoints[1:]:
            delta = self.shortest_rotation_delta(prev, wp)
            abs_delta = np.abs(delta)
            safe_rate = np.where(self._ramp_rate > 1e-12, self._ramp_rate, 1.0)
            ramp_time = max(np.max(abs_delta / safe_rate), self.dt)
            segments.append((ramp_time, prev.copy(), delta))
            prev = prev + delta  # unwrapped end position
            segments.append((self._dwell, prev.copy(), np.zeros_like(prev)))
        return segments

    def step(self, t: float) -> np.ndarray:
        elapsed = 0.0
        for duration, start, delta in self._segments:
            if t < elapsed + duration:
                alpha = (t - elapsed) / duration if duration > 0 else 1.0
                alpha = np.clip(alpha, 0.0, 1.0)
                return self.wrap_rotation(start + alpha * delta)
            elapsed += duration
        if self._waypoints:
            return self.wrap_rotation(self._waypoints[-1])
        return self.joint_lower.copy()

    def is_done(self, t: float) -> bool:
        return t >= self._total_time
