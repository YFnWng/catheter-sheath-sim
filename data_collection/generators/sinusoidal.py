"""Sinusoidal generator — incommensurate-frequency excitation."""
from __future__ import annotations

import numpy as np

from .base import InputGenerator


class SinusoidalGenerator(InputGenerator):
    """Uncorrelated sinusoidal excitation with speed-aware frequency selection.

    Each joint oscillates at a random frequency derived from its actuator
    speed limit so the commanded rate never exceeds what the system can
    track.  Amplitude and center are re-randomized every segment so that
    turning points occur throughout the joint range, not just at the limits.

    Parameters
    ----------
    joint_lower_limits, joint_upper_limits:
        Length-3 arrays [insertion, rotation, cable] bounds.
    dt:
        Simulation timestep (seconds).
    joint_max_speeds:
        (3,) max actuator speeds [insertion m/s, rotation deg/s, cable N/s
        or m/s].  Used to derive per-joint max frequency.  If None, falls
        back to ``frequencies``.
    speed_factor:
        Multiplier on the speed-derived f_max.  Values > 1 let the
        generator occasionally exceed actuator speed, covering more
        high-speed regions near range limits.
    frequencies:
        (3,) explicit base Hz (only used when ``joint_max_speeds`` is None).
    duration:
        Total trajectory duration in seconds.
    ramp_duration:
        Seconds to linearly ramp from zero to the sinusoidal waveform.
    freq_change_interval:
        Seconds between random frequency/amplitude/offset changes.
    freq_range:
        (low, high) multiplier on per-joint max frequency when drawing
        random frequencies each segment.
    amp_range:
        (low, high) fraction of half-range used when randomizing amplitude.
        1.0 = full swing to joint limits, 0.2 = small oscillation.
    limit_extension:
        Fraction of range to extend beyond joint limits for the internal
        sinusoid.  The output is still clamped to actual limits.  E.g.
        0.1 extends each side by 10% of the range, so the wave spends
        more time near the actual limits instead of always turning before
        reaching them.
    seed:
        Random seed for reproducibility.  None = random.
    """

    def __init__(
        self,
        joint_lower_limits: np.ndarray,
        joint_upper_limits: np.ndarray,
        dt: float,
        joint_max_speeds: np.ndarray = None,
        speed_factor: float = 1.3,
        frequencies: tuple = (0.10, 0.17, 0.31),
        duration: float = 60.0,
        ramp_duration: float = 2.0,
        freq_change_interval: float = 10.0,
        freq_range: tuple = (0.3, 1.0),
        amp_range: tuple = (0.2, 1.0),
        limit_extension: float = 0.1,
        seed: int = None,
    ) -> None:
        super().__init__(joint_lower_limits, joint_upper_limits, dt)
        self._duration = duration
        self._ramp_dur = ramp_duration
        self._freq_interval = freq_change_interval
        self._freq_range = freq_range
        self._amp_range = amp_range
        self._rng = np.random.default_rng(seed)

        full_range = self.joint_upper - self.joint_lower
        self._gen_lower = self.joint_lower - limit_extension * full_range
        self._gen_upper = self.joint_upper + limit_extension * full_range
        self._half_range = 0.5 * (self._gen_upper - self._gen_lower)

        if joint_max_speeds is not None:
            speeds = np.asarray(joint_max_speeds, dtype=float)
            self._f_max = speed_factor * speeds / (
                2.0 * np.pi * np.maximum(self._half_range, 1e-12))
        else:
            self._f_max = np.asarray(frequencies, dtype=float)

        self._schedule = self._build_schedule()

    def _build_schedule(self):
        """Pre-build list of (t_start, freq, phase_offset, center, amp).

        Phase offsets are chosen so the wave value is continuous across
        segment boundaries.
        """
        schedule = []
        t = 0.0
        prev_val = None
        while t < self._duration:
            freq = self._f_max * self._rng.uniform(
                self._freq_range[0], self._freq_range[1], size=3)
            amp_frac = self._rng.uniform(
                self._amp_range[0], self._amp_range[1], size=3)
            amp = self._half_range * amp_frac
            center_lo = self._gen_lower + amp
            center_hi = self._gen_upper - amp

            if prev_val is None:
                center = self._rng.uniform(center_lo, center_hi)
                phase = np.zeros(3)
            else:
                # Constrain center so prev_val is reachable: |prev_val - center| <= amp
                center_lo = np.maximum(center_lo, prev_val - amp)
                center_hi = np.minimum(center_hi, prev_val + amp)
                center = self._rng.uniform(center_lo, center_hi)
                # Solve: center + amp * sin(phase) = prev_val
                ratio = (prev_val - center) / np.maximum(amp, 1e-12)
                base_phase = np.arcsin(ratio)
                descend = self._rng.random(3) < 0.5
                phase = np.where(descend, np.pi - base_phase, base_phase)

            schedule.append((t, freq.copy(), phase.copy(),
                             center.copy(), amp.copy()))

            seg_end = min(t + self._freq_interval, self._duration)
            end_phase = phase + 2.0 * np.pi * freq * (seg_end - t)
            prev_val = center + amp * np.sin(end_phase)
            t = seg_end
        return schedule

    def _get_segment(self, t):
        seg = self._schedule[0]
        for s in self._schedule:
            if s[0] <= t:
                seg = s
            else:
                break
        t_start, freq, phase_offset, center, amp = seg
        phase = phase_offset + 2.0 * np.pi * freq * (t - t_start)
        return center, amp, phase

    def step(self, t: float) -> np.ndarray:
        center, amp, phase = self._get_segment(t)
        wave = center + amp * np.sin(phase)

        if t < self._ramp_dur:
            alpha = t / self._ramp_dur
            wave = alpha * wave

        wave = np.clip(wave, self.joint_lower, self.joint_upper)
        return self.wrap_rotation(wave)

    def is_done(self, t: float) -> bool:
        return t >= self._duration
