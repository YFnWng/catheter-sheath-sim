"""SofaWriter: writes estimated/display state to SOFA display nodes.

Provides ``add_estimation_display`` to create a ghost-catheter scene
graph (shape MO + sensor MO + OglModel visual line).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


class SofaWriter:
    """Writes node positions and sensor marker positions to SOFA display MOs.

    Parameters
    ----------
    shape_mo:
        ``Vec3d`` MechanicalObject with exactly *n_nodes* rows.
    sensor_mo:
        ``Vec3d`` MechanicalObject with *n_sensors* rows, or None.
    """

    def __init__(self, shape_mo, sensor_mo=None) -> None:
        self._shape_mo = shape_mo
        self._sensor_mo = sensor_mo

    def update_positions(self, node_positions: np.ndarray,
                         sensor_positions: np.ndarray = None) -> None:
        """Push node and sensor positions to SOFA display MOs.

        Parameters
        ----------
        node_positions : (n_nodes, 3) array
        sensor_positions : (n_sensors, 3) array or None
        """
        if self._shape_mo is not None and node_positions is not None:
            self._shape_mo.position.value = np.asarray(node_positions, dtype=float)

        if self._sensor_mo is None or sensor_positions is None:
            return
        arr = np.asarray(sensor_positions, dtype=float)
        capacity = len(self._sensor_mo.position.value)
        n = min(len(arr), capacity)
        sensor_pos = np.array(self._sensor_mo.position.value)
        sensor_pos[:n] = arr[:n]
        self._sensor_mo.position.value = sensor_pos


def add_estimation_display(
    root,
    n_nodes: int,
    ds: float,
    n_sensors: int,
    base_position: Sequence[float],
    base_orientation: Sequence[float],
) -> "SofaWriter":
    """Add ghost-catheter display nodes and return a :class:`SofaWriter`.

    Creates bare ``Vec3d`` MechanicalObjects with no OdeSolver so that
    ``FreeMotionAnimationLoop`` skips them during physics integration.

    Parameters
    ----------
    root:
        SOFA root node to attach the display subtree to.
    n_nodes:
        Number of nodes (= n_sections + 1).
    ds:
        Section length (m).
    n_sensors:
        Number of sensor markers to allocate (0 → no sensor MO).
    base_position:
        [x, y, z] of the catheter base in world frame.
    base_orientation:
        [qx, qy, qz, qw] quaternion of the catheter base.
    """
    from scipy.spatial.transform import Rotation as R_scipy

    base_rot = R_scipy.from_quat(base_orientation)
    local_z = base_rot.apply([0.0, 0.0, 1.0])
    bp = np.asarray(base_position, dtype=float)
    init_shape = [(bp + local_z * (k * ds)).tolist() for k in range(n_nodes)]

    disp = root.addChild("EstimatorDisplay")

    shape_mo = None
    if n_nodes > 0:
        shape_node = disp.addChild("EstimatedShape")
        shape_mo = shape_node.addObject(
            "MechanicalObject", name="ShapeMO", template="Vec3d",
            position=init_shape,
            showObject=True, showObjectScale=3.0,
            showColor=[0.0, 1.0, 0.5, 1.0],
        )
        shape_node.addObject(
            "EdgeSetTopologyContainer",
            edges=[[k, k + 1] for k in range(n_nodes - 1)],
        )
        shape_visual = shape_node.addChild("Visual")
        shape_visual.addObject(
            "OglModel", name="EstimatedLine",
            color=[0.0, 1.0, 0.5, 1.0],
            edges=[[k, k + 1] for k in range(n_nodes - 1)],
        )
        shape_visual.addObject(
            "IdentityMapping",
            input="@../ShapeMO", output="@EstimatedLine",
        )

    sensor_mo = None
    if n_sensors > 0:
        sensor_node = disp.addChild("SensorMarkers")
        sensor_mo = sensor_node.addObject(
            "MechanicalObject", name="SensorMO", template="Vec3d",
            position=init_shape[:n_sensors],
            showObject=True, showObjectScale=5.0,
            showColor=[1.0, 1.0, 0.2, 1.0],
        )

    return SofaWriter(shape_mo, sensor_mo)
