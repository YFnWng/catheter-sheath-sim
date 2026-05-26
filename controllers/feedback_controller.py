"""FeedbackController — SOFA bridge that orchestrates an observer and
a controller from the ``control/`` repo each simulation timestep.

Handles:
  - Reading SOFA state via SofaReader
  - Running the observer (EKF/UKF/GRU) for state estimation
  - Calling the controller with raw tip xyz measurement
  - Applying commands to SOFA (base rest_position + cable constraint)
  - Recording trajectories to HDF5 (same schema as DataCollectorController)
  - AdapJ motor babbling initialization phase

Design: The controller receives the *raw measurement* (tip xyz from SOFA)
for computing joint commands. The observation model is only used by the
observer for filtering / by MPPI for internal rollouts — never to
reconstruct the controller input.
"""
from __future__ import annotations

import os
import time as _time

import numpy as np
from scipy.spatial.transform import Rotation as R

import Sofa
import Sofa.Core

from data_collection.schema import TrajectoryRecord, write_hdf5


class FeedbackController(Sofa.Core.Controller):
    """SOFA controller that bridges observer + control algorithm.

    Parameters (passed as kwargs)
    ----------
    controller : control.base.BaseController
        Control algorithm instance (PID, MPPI, AdapJ, OpenLoop).
    observer : BaseObserver or None
        State estimator (EKF/UKF/GRU). Updated each step for filtering;
        its latent state is logged but not used as controller input.
    observation_model : callable(state) -> obs or None
        Maps observer state [z, zdot] to observation vector. Used only
        by MPPI for internal rollouts, not for controller input.
    reader : SofaReader
        Reads simulation state each timestep.
    reference : control.reference.ReferenceTrajectory
        Target trajectory.
    base_mechanical_object : SOFA MechanicalObject
        Rigid3d base DOF node (robot.base_mo).
    cable_constraint : SOFA object or None
        Cable constraint for writing cable commands.
    direction, base_position, base_orientation, prefab_rotation_offset :
        Same semantics as DataCollectorController.
    initial_base_pose : (7,) array or None
    initial_strain_coords : (n_sec, 3) array or None
    strain_mechanical_object : SOFA MO or None
    insertion_offset : float
    output_path : str
        HDF5 output path.
    record : bool
        If False, skip all H5 recording.
    warmup_steps : int
        Steps before engaging the controller.
    control_rate : int
        Run controller every N simulation steps. Hold last command between.
    metadata : dict
        Extra metadata for HDF5.
    babble_steps : int
        Motor babbling steps for AdapJ initialization (0 = skip).
    babble_amplitude : (n_joints,) array
        Per-joint babble range for AdapJ.
    duration : float
        Maximum simulation duration in seconds (0 = unlimited).
    """

    _SOFA_CABLE_SCALE = 0.01

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._controller = kwargs.pop("controller")
        self._observer = kwargs.pop("observer", None)
        self._observation_model = kwargs.pop("observation_model", None)
        self._reader = kwargs.pop("reader")
        self._reference = kwargs.pop("reference")

        self._base_mo = kwargs.pop("base_mechanical_object")
        self._cable_constraint = kwargs.pop("cable_constraint", None)
        self._output_path: str = kwargs.pop("output_path", "feedback.h5")
        self._record_enabled: bool = kwargs.pop("record", True)
        self._warmup_steps: int = kwargs.pop("warmup_steps", 50)
        self._control_rate: int = kwargs.pop("control_rate", 1)
        self._metadata: dict = kwargs.pop("metadata", {})
        self._duration: float = kwargs.pop("duration", 0.0)

        # Initial state
        init_bp = kwargs.pop("initial_base_pose", None)
        self._initial_base_pose = (
            np.asarray(init_bp, dtype=float) if init_bp is not None else None)
        init_sc = kwargs.pop("initial_strain_coords", None)
        self._initial_strain_coords = (
            np.asarray(init_sc, dtype=float) if init_sc is not None else None)
        self._strain_mo = kwargs.pop("strain_mechanical_object", None)
        self._insertion_offset = float(kwargs.pop("insertion_offset", 0.0))

        # Base kinematics
        self._base_home_position = np.asarray(
            kwargs.pop("base_position", np.zeros(3)), dtype=float)
        base_orient_quat = np.asarray(
            kwargs.pop("base_orientation", [0.0, 0.0, 0.0, 1.0]), dtype=float)
        self._base_home_orientation = R.from_quat(base_orient_quat)
        prefab_rot_quat = np.asarray(
            kwargs.pop("prefab_rotation_offset", [0.0, 0.0, 0.0, 1.0]),
            dtype=float)
        self._prefab_rot_offset = R.from_quat(prefab_rot_quat)
        direction = np.asarray(
            kwargs.pop("direction", [0.0, 0.0, 1.0]), dtype=float)
        direction = direction / np.linalg.norm(direction)
        self._direction = direction @ self._base_home_orientation.as_matrix().T

        self._cable_data = None
        if self._cable_constraint is not None:
            self._cable_data = self._cable_constraint.findData("value")

        # AdapJ babbling config
        self._babble_steps = int(kwargs.pop("babble_steps", 0))
        babble_amp = kwargs.pop("babble_amplitude", None)
        self._babble_amplitude = (
            np.asarray(babble_amp, dtype=float)
            if babble_amp is not None else None)

        # State
        self._record = TrajectoryRecord()
        self._control_records = {
            "reference": [],
            "compute_time": [],
            "update_time": [],
            "latent_state": [],
        }
        self._step = -1  # incremented to 0 on first call
        self._done = False
        self._joint_cmd = np.zeros(3)
        self._last_compute_time = 0.0
        self._last_update_time = 0.0
        self._in_control_phase = False
        self._last_sofa_gt = None  # cached from previous onAnimateEnd
        self._last_tip_xyz = None

        # Babbling state
        self._babbling = self._babble_steps > 0
        self._babble_states = []
        self._babble_actuations = []
        self._babble_count = 0

    # ------------------------------------------------------------------
    # SOFA callbacks
    # ------------------------------------------------------------------

    def onAnimateBeginEvent(self, _event) -> None:
        if self._done:
            return
        if self._base_mo is None or len(self._base_mo.position.value) == 0:
            return

        self._step += 1
        self._in_control_phase = False

        # Step 0: apply initial state, record in onAnimateEnd with t=0
        if self._step == 0:
            self._apply_initial_state()
            return
        
        dt = float(self._base_mo.getContext().dt.value)
        t = (self._step - 1) * dt

        # Check duration
        if self._duration > 0 and t > self._duration:
            self._finish()
            return

        # Warmup: let transients settle
        if self._step <= self._warmup_steps:
            return

        # Use tip xyz cached from previous onAnimateEnd
        tip_xyz = self._last_tip_xyz

        # --- Babbling phase (AdapJ) ---
        if self._babbling:
            self._run_babble_step(tip_xyz, t)
            return

        # --- Control phase ---
        ctrl_step = self._step - self._warmup_steps
        if ctrl_step % self._control_rate != 0:
            # Between control ticks: hold last command
            self._apply_joint_commands(self._joint_cmd)
            return

        self._in_control_phase = True

        # Get reference at current time
        ref = self._reference.at(t)

        # Compute control using raw measurement
        t0 = _time.perf_counter()
        self._joint_cmd = self._controller.compute(tip_xyz, ref, t)
        self._last_compute_time = _time.perf_counter() - t0

        self._apply_joint_commands(self._joint_cmd)

    def onAnimateEndEvent(self, _event) -> None:
        if self._done:
            return

        sofa_gt = self._reader.read()
        self._last_sofa_gt = sofa_gt
        self._last_tip_xyz = sofa_gt.frame_poses[-1][:3].copy()

        dt = float(self._base_mo.getContext().dt.value)
        t = self._step * dt

        # Observer update
        if self._observer is not None and self._step > self._warmup_steps:
            y = self._build_observation(sofa_gt)
            self._observer.step(self._joint_cmd, y)

        # AdapJ online update
        self._last_update_time = 0.0
        if hasattr(self._controller, "update") and not self._babbling:
            if self._step > self._warmup_steps:
                t0 = _time.perf_counter()
                self._controller.update(self._last_tip_xyz, self._joint_cmd)
                self._last_update_time = _time.perf_counter() - t0

        # --- Recording ---
        if not self._record_enabled:
            return

        self._record.append(
            t=t,
            frame_poses=sofa_gt.frame_poses,
            strain_coords=sofa_gt.strain_coords,
            joint_commands=self._joint_cmd,
            contact_force_body=sofa_gt.contact_force_body,
            tip_force=np.zeros(3),
            frame_velocity=sofa_gt.frame_velocity,
            strain_velocity=sofa_gt.strain_velocity,
        )

        if self._in_control_phase:
            ref = self._reference.at(t)
            self._control_records["reference"].append(ref.copy())
            self._control_records["compute_time"].append(self._last_compute_time)
            self._control_records["update_time"].append(self._last_update_time)

            if self._observer is not None:
                self._control_records["latent_state"].append(
                    self._observer.state.copy())

    # ------------------------------------------------------------------
    # Babbling (AdapJ initialization)
    # ------------------------------------------------------------------

    def _run_babble_step(self, tip_xyz: np.ndarray, t: float) -> None:
        """Execute one motor babbling step for AdapJ initialization."""
        self._babble_count += 1

        # Record state
        self._babble_states.append(tip_xyz.copy())

        # Random actuation
        if self._babble_amplitude is not None:
            amp = self._babble_amplitude
        else:
            amp = (self._controller.joint_upper -
                   self._controller.joint_lower) * 0.2
        cmd = np.random.uniform(-amp, amp)
        cmd = np.clip(cmd, self._controller.joint_lower,
                      self._controller.joint_upper)
        self._babble_actuations.append(cmd.copy())
        self._joint_cmd = cmd
        self._apply_joint_commands(cmd)

        # Check if babbling is done
        if self._babble_count >= self._babble_steps:
            self._babbling = False

            states = np.array(self._babble_states)
            actuations = np.array(self._babble_actuations)
            self._controller.initialize(states, actuations)
            print(f"[FeedbackController] AdapJ initialized with "
                  f"{len(states)} babbling samples")

    # ------------------------------------------------------------------
    # State reading
    # ------------------------------------------------------------------

    def _build_observation(self, sofa_gt) -> np.ndarray:
        """Build observation vector for the observer from SOFA ground truth.

        Constructs [xi, eta] from frame poses and velocities,
        matching the format expected by the latent observer.
        """
        from state_estimation.utils import SE3_to_R9, world_velocity_to_body
        from state_estimation.utils import pose_from_vec7

        tip_pose = sofa_gt.frame_poses[-1]
        T_tip = pose_from_vec7(tip_pose)
        if hasattr(self, "_X_ref"):
            T_rel = self._X_ref.between(T_tip)
        else:
            T_rel = T_tip
        xi = SE3_to_R9(T_rel)

        if sofa_gt.frame_velocity is not None and len(sofa_gt.frame_velocity) > 0:
            tip_vel_world = sofa_gt.frame_velocity[-1]
            eta = world_velocity_to_body(tip_pose, tip_vel_world)
        else:
            eta = np.zeros(6)

        return np.concatenate([xi, eta]).astype(np.float32)

    # ------------------------------------------------------------------
    # Joint command application (same as DataCollectorController)
    # ------------------------------------------------------------------

    def _apply_initial_state(self) -> None:
        """Write saved base_pose and strain_coords to SOFA MOs."""
        if self._initial_base_pose is not None:
            bp = self._initial_base_pose
            with self._base_mo.position.writeable() as pos:
                pos[0][:] = bp
            if (hasattr(self._base_mo, "rest_position")
                    and len(self._base_mo.rest_position.value) > 0):
                with self._base_mo.rest_position.writeable() as rest:
                    rest[0][:] = bp

        if self._initial_strain_coords is not None and self._strain_mo is not None:
            sc = self._initial_strain_coords
            with self._strain_mo.position.writeable() as strain_pos:
                n = min(len(sc), len(strain_pos))
                for i in range(n):
                    strain_pos[i][:] = sc[i]
            if hasattr(self._strain_mo, "rest_position"):
                with self._strain_mo.rest_position.writeable() as rest:
                    n = min(len(sc), len(rest))
                    for i in range(n):
                        rest[i][:] = sc[i]

    def _apply_joint_commands(self, joint_cmd: np.ndarray) -> None:
        """Apply [insertion, rotation, cable] to SOFA objects."""
        insertion = joint_cmd[0] + self._insertion_offset
        rotation_deg = joint_cmd[1]
        cable_val = joint_cmd[2]

        translation = self._direction * insertion
        rotation = R.from_rotvec(rotation_deg * self._direction, degrees=True)
        base_orientation = (
            rotation * self._base_home_orientation * self._prefab_rot_offset
        ).as_quat()
        target_pos = (self._base_home_position + translation).tolist()
        target_ori = base_orientation.tolist()

        if (hasattr(self._base_mo, "rest_position")
                and len(self._base_mo.rest_position.value) > 0):
            with self._base_mo.rest_position.writeable() as rest:
                rest[0][0:3] = target_pos
                rest[0][3:7] = target_ori

        if self._cable_data is not None:
            self._cable_data.value = [cable_val * self._SOFA_CABLE_SCALE]

    # ------------------------------------------------------------------
    # Finish and save
    # ------------------------------------------------------------------

    def _finish(self) -> None:
        self._done = True
        if not self._record_enabled:
            print("[FeedbackController] Done (recording disabled).")
            return

        n = len(self._record.timestamps)
        print(f"[FeedbackController] Done: {n} timesteps recorded.")
        os.makedirs(os.path.dirname(os.path.abspath(self._output_path)),
                    exist_ok=True)
        meta = {
            "controller": type(self._controller).__name__,
            "n_timesteps": n,
            **self._metadata,
        }

        write_hdf5(self._output_path, self._record, metadata=meta,
                    extra=self._control_records)
        print(f"[FeedbackController] Saved to {self._output_path}")

    @property
    def done(self) -> bool:
        return self._done
