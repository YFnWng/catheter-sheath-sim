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

# ---------------------------------------------------------------------------
# Catheter parameters
# ---------------------------------------------------------------------------
CATHETER_LENGTH = 160.0          # mm
CATHETER_SECTIONS = 32
CATHETER_FRAMES = 64
CATHETER_RADIUS = 1.45           # mm
CATHETER_MASS = 0.04             # kg
CATHETER_YOUNG_MODULUS = 8.0e5   # Pa
CATHETER_POISSON = 0.38
CATHETER_RAYLEIGH = 0.05

# In Cosserat prefab, the rod is along the local X axis. We use the convention that 
# the rod is along the local Z axis. So there are two rotations applied to the prefab:
# 1) Rotate -90 degrees about local Y to align rod with local Z.
# 2) A global rotation to set the home orientation of the catheter.
CATHETER_BASE_POSITION = [0.0, 0.0, -CATHETER_LENGTH]
CATHETER_BASE_ORIENTATION = R.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True)
PREFAB_ROTATION_OFFSET = R.from_euler("xyz", [0.0, -90.0, 0.0], degrees=True)
CATHETER_INSERTION_DIRECTION = [0.0, 0.0, 1.0] # In local frame
CATHETER_INSERTION_SPEED = 30.0  # mm/s
CATHETER_MAX_TRAVEL = 160.0      # mm
CATHETER_ROTATION_SPEED = 30.0   # deg/s
CATHETER_MAX_ROTATION = 180.0    # degrees

CABLE_OFFSET = 1.4               # mm lateral offset from centreline
CABLE_POINT_COUNT = 16
CABLE_PULL_INCREMENT = 3.0       # mm/s
CABLE_PULL_MIN = 0.0
CABLE_PULL_MAX = 30.0


class CatheterRobot:
    """Tendon-driven Cosserat rod catheter with a single actuation cable.

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
    insertion_direction : list[float]
        Unit vector along which the catheter is inserted.
    joint_rate : list[float]
        Rates for [insertion (mm/s), rotation (deg/s), cable (mm/s)].
    joint_upper_limits : list[float]
        Upper bounds for [insertion, rotation, cable].
    joint_lower_limits : list[float]
        Lower bounds for [insertion, rotation, cable].
    point_collision_model_path : str
        SOFA link path to the catheter PointCollisionModel.
    """

    def __init__(self, root: Sofa.Core.Node) -> None:
        self._prefab, self._collision, self.cable_constraint = _build(root)
        self.base_mo = self._prefab.rigidBaseNode.RigidBaseMO  # type: ignore[attr-defined]

        self.base_position: List[float] = list(CATHETER_BASE_POSITION)
        self.base_orientation: List[float] = list(CATHETER_BASE_ORIENTATION.as_quat().tolist())
        self.prefab_rotation_offset: List[float]  = PREFAB_ROTATION_OFFSET.as_quat().tolist()
        self.insertion_direction: List[float] = list(CATHETER_INSERTION_DIRECTION)
        self.joint_rate: List[float] = [
            CATHETER_INSERTION_SPEED,
            CATHETER_ROTATION_SPEED,
            CABLE_PULL_INCREMENT,
        ]
        self.joint_upper_limits: List[float] = [
            CATHETER_MAX_TRAVEL,
            CATHETER_MAX_ROTATION,
            CABLE_PULL_MAX,
        ]
        self.joint_lower_limits: List[float] = [
            0.0,
            -CATHETER_MAX_ROTATION,
            CABLE_PULL_MIN,
        ]

    @property
    def point_collision_model_path(self) -> str:
        return self._collision.PointCollisionModel.getLinkPath()


# ---------------------------------------------------------------------------
# Internal construction helpers
# ---------------------------------------------------------------------------

def _build(
    root: Sofa.Core.Node,
) -> Tuple[CosseratBase, Sofa.Core.Node, Optional[Sofa.Core.Object]]:
    solver_node = root.addChild("CatheterSimulation")
    solver_node.addObject(
        "EulerImplicitSolver",
        rayleighStiffness=CATHETER_RAYLEIGH,
        rayleighMass=1e-3,
    )
    solver_node.addObject(
        "SparseLDLSolver",
        name="solver",
        template="CompressedRowSparseMatrixd",
    )
    solver_node.addObject("GenericConstraintCorrection")

    params = _build_params()
    # CosseratBase is a Sofa.Prefab: passing parent= registers it as a child
    # of solver_node automatically.  Do NOT also wrap in addChild() — that
    # would add the same node twice and trigger a SceneCheckDuplicatedName warning.
    # CosseratBase rotation= expects Euler angles in degrees (3 elements),
    # not a quaternion.  Passing a 4-element quaternion corrupts the heap.
    prefab_base_rotation = (CATHETER_BASE_ORIENTATION * PREFAB_ROTATION_OFFSET).as_euler("xyz", degrees=True).tolist()
    prefab = CosseratBase(
        parent=solver_node,
        params=params,
        name="catheter",
        translation=CATHETER_BASE_POSITION,
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

    cable_constraint = _add_cable(prefab)
    return prefab, collision, cable_constraint


def _add_cable(prefab: CosseratBase) -> Optional[Sofa.Core.Object]:
    frame_states = np.asarray(prefab.frames3D, dtype=float)
    cable_points = compute_cable_points(frame_states, CABLE_POINT_COUNT, CABLE_OFFSET)
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
        valueType="displacement",
    )

    cable_mo = getattr(cable, "MechanicalObject", None)
    if cable_mo is not None:
        cable_mo.showObject = True
        cable_mo.showIndices = False
        cable_mo.showObjectScale = 1.25
        cable_mo.showColor = [0.2, 0.85, 0.3, 1.0]

    return getattr(cable, "CableConstraint", None)


def _build_params() -> Parameters:
    geometry = BeamGeometryParameters(
        beam_length=CATHETER_LENGTH,
        nb_section=CATHETER_SECTIONS,
        nb_frames=CATHETER_FRAMES,
        build_collision_model=1,
    )
    physics = BeamPhysicsParametersNoInertia(
        beam_mass=CATHETER_MASS,
        young_modulus=CATHETER_YOUNG_MODULUS,
        poisson_ratio=CATHETER_POISSON,
        beam_radius=CATHETER_RADIUS,
        beam_length=CATHETER_LENGTH,
    )
    params = Parameters(beam_geo_params=geometry, beam_physics_params=physics)
    params.simu_params.rayleigh_stiffness = CATHETER_RAYLEIGH
    params.simu_params.rayleigh_mass = 1e-3
    return params
