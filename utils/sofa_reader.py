"""SofaReader: extracts raw simulation state from SOFA scene objects.

Convention note
---------------
SOFA's Cosserat plugin uses **local X = rod tangent** for each frame.
The estimation core uses **local Z = rod tangent**.

Frame and base **positions** are returned as-is (world frame).
Frame and base **orientations** are right-multiplied by R_prefab^{-1}
to convert to the estimation's local-Z convention when
``prefab_rotation_offset`` is provided.
Strain coordinates are rotated by R_prefab to the local-Z convention.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from .sofa_data import SofaGroundTruth


class SofaReader:
    """Reads Cosserat rod state from a SOFA scene.

    Parameters
    ----------
    prefab:
        The ``CosseratBase`` prefab node.
    base_mo:
        The ``RigidBaseMO`` MechanicalObject.
    cable_constraints:
        List of ``CableConstraint`` SOFA objects.
    prefab_rotation_offset:
        Quaternion [qx, qy, qz, qw] aligning the Cosserat prefab rod axis
        with the scene convention.  Pass ``None`` if no convention difference.
    cable_mode:
        ``"displacement"`` or ``"force"``.
    constraint_solver:
        The scene's constraint solver (needs ``computeConstraintForces=True``).
    contact_listener:
        A ``ContactListener`` SOFA object (for contact point queries).
    """

    def __init__(self, prefab, base_mo,
                 cable_constraints=None,
                 prefab_rotation_offset=None,
                 cable_mode: str = "displacement",
                 constraint_solver=None,
                 contact_listener=None) -> None:
        self._prefab = prefab
        self._base_mo = base_mo
        self._cable_mode = cable_mode
        self._constraint_solver = constraint_solver
        self._contact_listener = contact_listener

        self._frame_mo = prefab.cosseratFrame.FramesMO

        _cc_list = [cc for cc in (cable_constraints or []) if cc is not None]
        self._cable_datas = [cc.findData("value") for cc in _cc_list]
        if prefab_rotation_offset is not None:
            rot = R.from_quat(prefab_rotation_offset)
            self._rot_inv = rot.inv()
            self._strain_rot = rot.as_matrix()
        else:
            self._rot_inv = None
            self._strain_rot = None

    def read(self) -> SofaGroundTruth:
        """Extract current state from SOFA and return a ``SofaGroundTruth``."""
        frame_poses = np.array(
            self._frame_mo.position.value,
            dtype=float,
        )

        strain_coords_sofa = np.array(
            self._prefab.cosseratCoordinate.cosseratCoordinateMO.position.value,
            dtype=float,
        )

        base_pose = np.array(
            self._base_mo.position.value[0],
            dtype=float,
        )

        cable_values = []
        for cd in self._cable_datas:
            val = cd.value
            cable_values.append(float(val[0]) if hasattr(val, "__len__") else float(val))

        if self._rot_inv is not None:
            frame_poses, base_pose = self._convert_orientations(frame_poses, base_pose)

        if self._strain_rot is not None:
            strain_coords = strain_coords_sofa @ self._strain_rot.T
        else:
            strain_coords = strain_coords_sofa

        if self._cable_mode == "displacement":
            cable_disp = cable_values[0] if cable_values else 0.0
            cable_tensions = None
        else:
            cable_disp = 0.0
            _SOFA_CABLE_SCALE_INV = 100.0
            if cable_values:
                cable_tensions = np.array(cable_values) * _SOFA_CABLE_SCALE_INV
            else:
                cable_tensions = None

        n_sections = len(strain_coords)
        contact_force_body = self._read_contact_forces(frame_poses, n_sections)

        return SofaGroundTruth(
            frame_poses=frame_poses,
            strain_coords=strain_coords,
            base_pose=base_pose,
            cable_disp=cable_disp,
            cable_tensions=cable_tensions,
            contact_force_body=contact_force_body,
        )

    def _read_contact_forces(self, frame_poses: np.ndarray, n_sections: int) -> np.ndarray:
        """Return per-node contact force in body frame, shape (n_nodes, 3).

        Reads the ``lambda`` data field on ``FramesMO``, which contains the
        per-DOF constraint reaction force in world frame (Rigid3d wrench:
        [fx, fy, fz, tx, ty, tz]).  The translational part [fx, fy, fz] is
        rotated into the rod body frame.

        This is more reliable than reconstructing forces from the solver's
        ``constraintForces`` vector, which is only intermittently populated.
        """
        n_nodes = n_sections + 1

        lam_data = self._frame_mo.getData("lambda")
        if lam_data is None:
            return np.zeros((n_nodes, 3))

        lam = np.array(lam_data.value, dtype=float)
        if len(lam) == 0:
            return np.zeros((n_nodes, 3))

        # lam is (n_frames, 6) or (n_frames, 7) for Rigid3d
        # Extract translational force [fx, fy, fz] (first 3 components of wrench)
        n = min(n_nodes, len(lam))
        result = np.zeros((n_nodes, 3))

        for k in range(n):
            f_world = np.array([lam[k][0], lam[k][1], lam[k][2]], dtype=float)
            if np.linalg.norm(f_world) < 1e-15:
                continue
            # Rotate to body frame
            R_k = R.from_quat(frame_poses[k, 3:7])
            result[k] = R_k.inv().apply(f_world)

        return result

    def _convert_orientations(self, frame_poses, base_pose):
        R_frames = R.from_quat(frame_poses[:, 3:7])
        frame_poses = frame_poses.copy()
        frame_poses[:, 3:7] = (R_frames * self._rot_inv).as_quat()

        R_base = R.from_quat(base_pose[3:7])
        base_pose = base_pose.copy()
        base_pose[3:7] = (R_base * self._rot_inv).as_quat()

        return frame_poses, base_pose
