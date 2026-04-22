"""PlotController — reads SOFA state each step and sends to a DiagnosticPlotter."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

import Sofa
import Sofa.Core

from utils.plotter import DiagnosticPlotter
from utils.sofa_reader import SofaReader


class PlotController(Sofa.Core.Controller):
    """Reads base pose, cable tension, and contact forces each step.

    Uses a ``SofaReader`` to extract the full simulation state, then
    packs it for the diagnostic plotter.

    Parameters (passed as kwargs)
    ----------
    plotter : DiagnosticPlotter
    reader : SofaReader
    n_nodes : int
    base_home_position : array-like
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._plotter: DiagnosticPlotter = kwargs.pop("plotter")
        self._reader: SofaReader = kwargs.pop("reader")
        self._n_nodes = kwargs.pop("n_nodes")
        self._base_home = np.asarray(
            kwargs.pop("base_home_position", [0.0, 0.0, 0.0]), dtype=float,
        )
        self._tip_force_field = kwargs.pop("tip_force_field", None)

    def onAnimateBeginEvent(self, _event):
        sofa_gt = self._reader.read()
        ctx = self._reader._base_mo.getContext()
        t = float(ctx.time.value)

        # Base pose (relative to home)
        base_pos = sofa_gt.base_pose[:3] - self._base_home
        base_rot_deg = R.from_quat(sofa_gt.base_pose[3:7]).as_euler("xyz", degrees=True)

        # Cable tensions
        if sofa_gt.cable_tensions is not None:
            cable_tensions = sofa_gt.cable_tensions
        else:
            cable_tensions = np.zeros(1)

        # External force (body frame, per node): contact + tip load
        gt_F = np.zeros((self._n_nodes, 3), dtype=np.float32)
        n = min(self._n_nodes, len(sofa_gt.contact_force_body))
        gt_F[:n] = sofa_gt.contact_force_body[:n]

        tip_force_3d = np.zeros(3)
        if self._tip_force_field is not None:
            forces = np.array(self._tip_force_field.findData("forces").value)
            if forces.size >= 3:
                tip_force_3d = forces.flat[:3].copy()
                # Add tip load to the last node in the spatial plot
                gt_F[-1] += tip_force_3d

        plot_data = {
            "t": t,
            "base_pos": base_pos,
            "base_rot": base_rot_deg,
            "cable_tensions": cable_tensions,
            "gt_F": gt_F,
            "tip_load": tip_force_3d,
        }

        self._plotter.send(plot_data)
