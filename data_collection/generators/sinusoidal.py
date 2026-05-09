"""Sinusoidal generator — incommensurate-frequency excitation."""
from __future__ import annotations

import numpy as np

from .base import InputGenerator


class SinusoidalGenerator(InputGenerator):
    """Uncorrelated sinusoidal excitation with ramp-in and random frequency changes.

    Each joint oscillates at a different frequency.  The trajectory begins
    at zero (joint_lower) and linearly ramps to the sinusoidal waveform
    over ``ramp_duration`` seconds.  Frequencies are randomly re-drawn
    every ``freq_change_interval`` seconds (if > 0) to prevent the network
    from memorizing periodic patterns.

    Parameters
    ----------
    joint_lower_limits, joint_upper_limits:
        Length-3 arrays [insertion, rotation, cable] bounds.
    dt:
        Simulation timestep (seconds).
    frequencies:
        (3,) base Hz for [insertion, rotation, cable].  Default uses
        incommensurate ratios (0.10, 0.17, 0.31).
    duration:
        Total trajectory duration in seconds.
    ramp_duration:
        Seconds to linearly ramp from zero to the sinusoidal waveform.
    freq_change_interval:
        Seconds between random frequency changes.  0 = fixed frequencies.
    freq_range:
        (low, high) multiplier on base frequencies when re-drawing.
    seed:
        Random seed for reproducibility.  None = random.
    """

    def __init__(
        self,
        joint_lower_limits: np.ndarray,
        joint_upper_limits: np.ndarray,
        dt: float,
        frequencies: tuple = (0.10, 0.17, 0.31),
        duration: float = 60.0,
        ramp_duration: float = 2.0,
        freq_change_interval: float = 10.0,
        freq_range: tuple = (0.5, 2.0),
        seed: int = None,
    ) -> None:
        super().__init__(joint_lower_limits, joint_upper_limits, dt)
        self._base_freq = np.asarray(frequencies, dtype=float)
        self._duration = duration
        self._ramp_dur = ramp_duration
        self._freq_interval = freq_change_interval
        self._freq_range = freq_range
        self._rng = np.random.default_rng(seed)

        self._mid = 0.5 * (self.joint_lower + self.joint_upper)
        self._amp = 0.5 * (self.joint_upper - self.joint_lower)

        # Build frequency schedule
        self._freq_schedule = self._build_schedule()

    def _build_schedule(self):
        """Pre-build list of (t_start, frequencies, phase_offset) segments."""
        if self._freq_interval <= 0:
            return [(0.0, self._base_freq.copy(), np.zeros(3))]

        schedule = []
        t = 0.0
        freq = self._base_freq.copy()
        phase = np.zeros(3)
        while t < self._duration:
            schedule.append((t, freq.copy(), phase.copy()))
            # Compute phase at end of this segment for continuity
            seg_end = min(t + self._freq_interval, self._duration)
            phase = phase + 2.0 * np.pi * freq * (seg_end - t)
            t = seg_end
            # Draw new frequencies
            freq = self._base_freq * self._rng.uniform(
                self._freq_range[0], self._freq_range[1], size=3)
        return schedule

    def _get_freq_and_phase(self, t):
        """Find the active frequency segment for time t."""
        # Binary search would be faster but schedule is short
        seg = self._freq_schedule[0]
        for s in self._freq_schedule:
            if s[0] <= t:
                seg = s
            else:
                break
        t_start, freq, phase_offset = seg
        return freq, phase_offset + 2.0 * np.pi * freq * (t - t_start)

    def step(self, t: float) -> np.ndarray:
        freq, phase = self._get_freq_and_phase(t)
        wave = self._mid + self._amp * np.sin(phase)

        # Ramp from zero command to wave
        if t < self._ramp_dur:
            alpha = t / self._ramp_dur
            zero = np.zeros_like(self.joint_lower)
            wave = zero + alpha * (wave - zero)

        return self.wrap_rotation(wave)

    def is_done(self, t: float) -> bool:
        return t >= self._duration
