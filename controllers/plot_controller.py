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
        self._fb_controller = kwargs.pop("feedback_controller", None)
        self._data_collector = kwargs.pop("data_collector", None)
        self._prev_est_rot_raw = None
        self._est_rot_unwrapped = 0.0

    def onAnimateBeginEvent(self, _event):
        sofa_gt = self._reader.read()
        ctx = self._reader._base_mo.getContext()
        t = float(ctx.time.value)

        # Base pose (relative to home)
        base_pos = sofa_gt.base_pose[:3] - self._base_home
        base_rot_deg = R.from_quat(sofa_gt.base_pose[3:7]).as_euler("xyz", degrees=True)

        # Command from joint_cmd (feedback controller or data collector)
        joint_cmd = np.zeros(3)
        if self._fb_controller is not None:
            joint_cmd = self._fb_controller._joint_cmd
        elif self._data_collector is not None:
            joint_cmd = self._data_collector._joint_cmd

        insertion_actual = float(np.linalg.norm(base_pos))
        rotation_actual = float(base_rot_deg[2])
        insertion_cmd = float(joint_cmd[0])
        rotation_cmd = float(joint_cmd[1])

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
                gt_F[-1] += tip_force_3d

        plot_data = {
            "t": t,
            "base_pos": np.array([insertion_actual]),
            "base_rot": np.array([rotation_actual]),
            "base_rest_pos": np.array([insertion_cmd]),
            "base_rest_rot": np.array([rotation_cmd]),
            "cable_tensions": cable_tensions,
            "gt_F": gt_F,
            "tip_load": tip_force_3d,
        }

        # Estimated base state from observer/rollout
        if self._fb_controller is not None:
            err = self._fb_controller._last_tracking_error
            plot_data["tracking_error"] = err if err is not None else np.zeros(3)

            # Extract estimated base insertion + rotation from latent state
            est_ins, est_rot = self._extract_estimated_base()
            if est_ins is not None:
                plot_data["base_est_pos"] = np.array([est_ins])
                plot_data["base_est_rot"] = np.array([est_rot])

        self._plotter.send(plot_data)

    def _extract_estimated_base(self):
        """Extract estimated insertion (m) and rotation (deg) from the
        feedback controller's current display state (observer or rollout)."""
        fb = self._fb_controller
        if fb is None:
            return None, None

        # Pick the same source as the ghost shape display
        source = getattr(fb, '_shape_source', 'observer')
        if source == 'rollout' and fb._rollout_state is not None:
            state = fb._rollout_state
        elif fb._observer is not None:
            state = fb._observer.state
        else:
            return None, None

        obs = fb._observation_model
        if obs is None:
            return None, None

        d = obs._d
        if len(state) <= 2 * d:
            return None, None  # no base state in latent vector

        # Encoded base position: [insertion, cos(rot), sin(rot)]
        base_enc = state[2 * d:2 * d + 3]
        insertion = float(base_enc[0])
        raw_deg = float(np.degrees(np.arctan2(base_enc[2], base_enc[1])))

        # Unwrap: arctan2 returns [-180, 180] but ground truth is cumulative
        if self._prev_est_rot_raw is not None:
            delta = raw_deg - self._prev_est_rot_raw
            if delta > 180:
                delta -= 360
            elif delta < -180:
                delta += 360
            self._est_rot_unwrapped += delta
        else:
            self._est_rot_unwrapped = raw_deg
        self._prev_est_rot_raw = raw_deg

        return insertion, self._est_rot_unwrapped
