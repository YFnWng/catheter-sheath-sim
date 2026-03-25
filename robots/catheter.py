from __future__ import annotations

import os
import yaml
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

import Sofa
import Sofa.Core

from cosserat.CosseratBase import CosseratBase  # type: ignore
from actuators.cable import PullingCable  # type: ignore
from useful.params import (  # type: ignore
    BeamGeometryParameters,
    BeamPhysicsParametersNoInertia,
    Parameters,
)

from utils.cable_utils import compute_cable_points

_DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "catheter_ablation.yaml",
)


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class CatheterRobot:
    """Tendon-driven Cosserat rod catheter with a single actuation cable.

    Parameters
    ----------
    root:
        The SOFA root node.
    config_path:
        Path to the catheter YAML config file.  Defaults to
        ``simulation/configs/catheter_ablation.yaml`` next to this file.
    cable_mode:
        ``"displacement"`` (default) or ``"force"`` — passed to
        ``PullingCable(valueType=...)``.

    After construction the following attributes are available for wiring up
    controllers and contact listeners:

    Attributes
    ----------
    base_mo : Sofa MechanicalObject
        Rigid3d base DOF node (RigidBaseMO).
    cable_constraint : Sofa object or None
        CableConstraint object for the actuation cable.
    base_position : list[float]
        Home position of the catheter base [x, y, z] (mm).
    base_orientation : list[float]
        Home orientation of the catheter base as quaternion [qx,qy,qz,qw].
    prefab_rotation_offset : list[float]
        Quaternion [qx,qy,qz,qw] of the rotation applied to align the Cosserat
        prefab local-X rod axis with the scene local-Z convention.
    insertion_direction : list[float]
        Unit vector along which the catheter is inserted (local frame).
    joint_rate : list[float]
        Rates for [insertion (mm/s), rotation (deg/s), cable (mm/s)].
    joint_upper_limits : list[float]
        Upper bounds for [insertion, rotation, cable].
    joint_lower_limits : list[float]
        Lower bounds for [insertion, rotation, cable].
    point_collision_model_path : str
        SOFA link path to the catheter PointCollisionModel.
    """

    def __init__(
        self,
        root: Sofa.Core.Node,
        config_path: str = _DEFAULT_CONFIG,
        cable_mode: str = "displacement",
    ) -> None:
        cfg = _load_config(config_path)
        robot_cfg = cfg.get("robot", {})
        act_cfg = cfg.get("actuation", {})
        ctrl_cfg = cfg.get("controller", {})

        self._prefab, self._collision, self.cable_constraint = _build(
            root, robot_cfg, act_cfg, cable_mode=cable_mode
        )
        self.base_mo = self._prefab.rigidBaseNode.RigidBaseMO  # type: ignore[attr-defined]

        self.base_position: List[float] = list(robot_cfg["base_position"])
        base_home_orientation = R.from_euler(
            "xyz",
            robot_cfg.get("base_orientation_euler_xyz_deg", [0.0, 0.0, 0.0]),
            degrees=True,
        )
        self.base_orientation: List[float] = base_home_orientation.as_quat().tolist()
        prefab_rotation = R.from_euler(
            "xyz",
            robot_cfg.get("prefab_rotation_euler_xyz_deg", [0.0, -90.0, 0.0]),
            degrees=True,
        )
        self.prefab_rotation_offset: List[float] = prefab_rotation.as_quat().tolist()
        self.insertion_direction: List[float] = list(
            ctrl_cfg.get("insertion_direction", [0.0, 0.0, 1.0])
        )
        self.joint_rate: List[float] = [
            float(ctrl_cfg.get("insertion_speed", 30.0)),
            float(ctrl_cfg.get("rotation_speed", 30.0)),
            float(act_cfg.get("pull_increment", 3.0)),
        ]
        self.joint_upper_limits: List[float] = [
            float(ctrl_cfg.get("max_travel", 160.0)),
            float(ctrl_cfg.get("max_rotation", 180.0)),
            float(act_cfg.get("pull_max", 30.0)),
        ]
        self.joint_lower_limits: List[float] = [
            0.0,
            -float(ctrl_cfg.get("max_rotation", 180.0)),
            float(act_cfg.get("pull_min", 0.0)),
        ]

    @property
    def point_collision_model_path(self) -> str:
        return self._collision.PointCollisionModel.getLinkPath()


# ---------------------------------------------------------------------------
# Internal construction helpers
# ---------------------------------------------------------------------------

def _build(
    root: Sofa.Core.Node,
    robot_cfg: dict,
    act_cfg: dict,
    cable_mode: str = "displacement",
) -> Tuple[CosseratBase, Sofa.Core.Node, Optional[Sofa.Core.Object]]:
    solver_node = root.addChild("CatheterSimulation")
    rayleigh = float(robot_cfg.get("rayleigh", 0.05))
    solver_node.addObject(
        "EulerImplicitSolver",
        rayleighStiffness=rayleigh,
        rayleighMass=1e-3,
    )
    solver_node.addObject(
        "SparseLDLSolver",
        name="solver",
        template="CompressedRowSparseMatrixd",
    )
    solver_node.addObject("GenericConstraintCorrection")

    params = _build_params(robot_cfg)

    base_pos = robot_cfg.get("base_position", [0.0, 0.0, -160.0])
    base_orient = R.from_euler(
        "xyz",
        robot_cfg.get("base_orientation_euler_xyz_deg", [0.0, 0.0, 0.0]),
        degrees=True,
    )
    prefab_rot = R.from_euler(
        "xyz",
        robot_cfg.get("prefab_rotation_euler_xyz_deg", [0.0, -90.0, 0.0]),
        degrees=True,
    )
    # CosseratBase rotation= expects Euler angles in degrees (3 elements),
    # not a quaternion.  Passing a 4-element quaternion corrupts the heap.
    prefab_base_rotation = (base_orient * prefab_rot).as_euler("xyz", degrees=True).tolist()
    prefab = CosseratBase(
        parent=solver_node,
        params=params,
        name="catheter",
        translation=base_pos,
        rotation=prefab_base_rotation,
    )

    prefab.rigidBaseNode.addObject(  # type: ignore[attr-defined]
        "RestShapeSpringsForceField",
        name="BaseAttachment",
        stiffness=1e8,
        angularStiffness=1e8,
        external_points=0,
        points=0,
        template="Rigid3d",
    )

    prefab.cosseratFrame.FramesMO.showObject = True  # type: ignore[attr-defined]
    prefab.cosseratFrame.FramesMO.showObjectScale = 0.8  # type: ignore[attr-defined]

    collision = prefab.addCollisionModel()
    if hasattr(collision, "CollisionDOFs"):
        collision.CollisionDOFs.showObject = False  # type: ignore[attr-defined]

    cable_constraint = _add_cable(prefab, act_cfg, cable_mode=cable_mode)
    return prefab, collision, cable_constraint


def _add_cable(
    prefab: CosseratBase,
    act_cfg: dict,
    cable_mode: str = "displacement",
) -> Optional[Sofa.Core.Object]:
    cable_offset = float(act_cfg.get("cable_offset", 1.4))
    cable_point_count = int(act_cfg.get("cable_point_count", 16))

    frame_states = np.asarray(prefab.frames3D, dtype=float)
    cable_points = compute_cable_points(frame_states, cable_point_count, cable_offset)
    if cable_points.shape[0] < 2:
        return None

    frame_node = prefab.rigidBaseNode.cosseratInSofaFrameNode  # type: ignore[attr-defined]
    attachment = frame_node.addChild("CableAttachment")
    guide_positions = cable_points.tolist()
    attachment.addObject(
        "MechanicalObject",
        name="CableGuideMO",
        template="Vec3d",
        position=guide_positions,
        showObject=True,
        showIndices=False,
        showObjectScale=0.8,
    )
    attachment.addObject("SkinningMapping", nbRef="1", name="CableSkinning")

    cable = PullingCable(
        attachedTo=attachment,
        name="ActuationCable",
        cableGeometry=guide_positions,
        valueType=cable_mode,
    )

    cable_mo = getattr(cable, "MechanicalObject", None)
    if cable_mo is not None:
        cable_mo.showObject = True
        cable_mo.showIndices = False
        cable_mo.showObjectScale = 1.25
        cable_mo.showColor = [0.2, 0.85, 0.3, 1.0]

    return getattr(cable, "CableConstraint", None)


def _build_params(robot_cfg: dict) -> Parameters:
    length = float(robot_cfg.get("length", 160.0))
    geometry = BeamGeometryParameters(
        beam_length=length,
        nb_section=int(robot_cfg.get("n_sections", 32)),
        nb_frames=int(robot_cfg.get("n_frames", 64)),
        build_collision_model=1,
    )
    physics = BeamPhysicsParametersNoInertia(
        beam_mass=float(robot_cfg.get("mass", 0.04)),
        young_modulus=float(robot_cfg.get("young_modulus", 8.0e5)),
        poisson_ratio=float(robot_cfg.get("poisson_ratio", 0.38)),
        beam_radius=float(robot_cfg.get("radius", 1.45)),
        beam_length=length,
    )
    params = Parameters(beam_geo_params=geometry, beam_physics_params=physics)
    params.simu_params.rayleigh_stiffness = float(robot_cfg.get("rayleigh", 0.05))
    params.simu_params.rayleigh_mass = 1e-3
    return params
