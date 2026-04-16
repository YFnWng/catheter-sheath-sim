from typing import Sequence

import numpy as np


def compute_cable_points(
    frame_states: Sequence[Sequence[float]],
    target_count: int,
    cable_location: Sequence[float] = (0.0, 0.0),
) -> np.ndarray:
    """Sample *target_count* points along a Cosserat frame sequence.

    Parameters
    ----------
    frame_states:
        Sequence of frame state vectors.  Only the first 3 components
        (x, y, z position) are used.
    target_count:
        Number of evenly-spaced arc-length samples to return.
    cable_location:
        2D offset ``[x, y]`` (mm) of the cable in the rod cross-section,
        added to the respective coordinates of every sample.

    Returns
    -------
    np.ndarray of shape (target_count, 3).
    """
    if target_count <= 0 or len(frame_states) == 0:
        return np.empty((0, 3))

    frames = np.asarray(frame_states, dtype=float)
    frames = frames.reshape((-1, frames.shape[-1]))
    if frames.shape[1] >= 3:
        positions = frames[:, :3].copy()
    else:
        positions = np.zeros((len(frames), 3))
        positions[:, : frames.shape[1]] = frames

    positions[:, 0] += float(cable_location[0])
    positions[:, 1] += float(cable_location[1])

    if target_count >= len(positions):
        return positions.copy()

    deltas = np.diff(positions, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    arc_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = arc_lengths[-1]
    if total_length <= 0.0:
        return np.repeat(positions[:1], target_count, axis=0)

    sample_distances = np.linspace(0.0, total_length, target_count)
    sampled = np.empty((target_count, 3))
    for axis in range(3):
        sampled[:, axis] = np.interp(sample_distances, arc_lengths, positions[:, axis])

    return sampled
