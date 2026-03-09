import math
import os
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

import Sofa
import Sofa.Core

SCENE_DIR = os.path.dirname(os.path.abspath(__file__))
HEART_MESH = os.path.join(SCENE_DIR, 
                          "../assets/full_heart_clean_reduced_transseptal.stl")

from cosserat.CosseratBase import CosseratBase  # type: ignore  # pylint: disable=import-error
from cosserat.usefulFunctions import pluginList as COSERAT_PLUGIN_LIST  # type: ignore
from actuators.cable import PullingCable  # type: ignore  # pylint: disable=import-error
from useful.params import (  # type: ignore  # pylint: disable=import-error
    BeamGeometryParameters,
    BeamPhysicsParametersNoInertia,
    Parameters,
)


# ---------------------------------------------------------------------------
# Scene-wide parameters
# ---------------------------------------------------------------------------
BACKGROUND_COLOR = [0.0, 0.0, 0.0, 1.0] #[0.05, 0.05, 0.08, 1.0]
DISPLAY_FLAGS = (
    "showVisualModels showBehaviorModels hideCollisionModels hideBoundingCollisionModels "
    "showForceFields hideInteractionForceFields hideWireframe showMechanicalMappings"
)

# Heart parameters (keeping STL units in millimeters)
HEART_POSITION = [60.0, -75.0, -20.0]
HEART_ORIENTATION = R.from_euler("xyz", [-55.0, -20.0, 0.0], degrees=True).as_quat()  # (x, y, z, w)

# Catheter Cosserat parameters (keeping STL units in millimeters)
CATHETER_LENGTH = 160.0  # mm
CATHETER_SECTIONS = 32
CATHETER_FRAMES = 64
CATHETER_RADIUS = 1.45  # mm
CATHETER_MASS = 0.04  # kg
CATHETER_YOUNG_MODULUS = 8.0e5  # Pa
CATHETER_POISSON = 0.38
CATHETER_RAYLEIGH = 0.05
CATHETER_BASE_TRANSLATION = [0.0, 0.0, -CATHETER_LENGTH]
CATHETER_BASE_ROTATION = [0.0, -90.0, 0.0]
CATHETER_INSERTION_SPEED = 30.0  # mm per second along +Z
CATHETER_MAX_TRAVEL = 160.0  # mm
CATHETER_BASE_POSITION = [0.0, 0.0, -CATHETER_LENGTH]
CATHETER_BASE_ORIENTATION = (
    R.from_euler("xyz", CATHETER_BASE_ROTATION, degrees=True).as_quat().tolist()
)
CATHETER_INSERTION_DIRECTION = [0.0, 0.0, 1.0]
CATHETER_ROTATION_SPEED = 30.0  # deg/s around the catheter axis
CATHETER_MAX_ROTATION = 180.0  # degrees of axial rotation allowed in either direction

CABLE_OFFSET = 1.4  # mm from the catheter centerline to the cable attachment points
CABLE_POINT_COUNT = 16
CABLE_PULL_INCREMENT = 3.0  # mm/s
CABLE_PULL_MIN = 0.0
CABLE_PULL_MAX = 30.0
CATHETER_ROTATION_SPEED = 30.0  # degrees per second around the catheter axis


class CatheterKeyboardController(Sofa.Core.Controller):
    """Keyboard control for insertion, axial rotation, and cable pulling."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.base_mo = kwargs.pop("base_mechanical_object", None)
        self.direction = self._normalize(kwargs.pop("direction", [0.0, 0.0, 1.0]))
        self.joint_pos = np.zeros(3)
        # self.joint_delta = np.zeros(3)
        self.joint_rate = kwargs.pop("joint_rate", np.zeros(3))
        self.joint_upper_limits = kwargs.pop("joint_upper_limits", np.zeros(3))
        self.joint_lower_limits = kwargs.pop("joint_lower_limits", np.zeros(3))
        self.base_home_position = np.asarray(
            kwargs.pop("base_position", np.zeros(3))
        )
        base_home_orientation = np.asarray(
            kwargs.pop("base_orientation", [0.0, 0.0, 0.0, 1.0])
        )
        self._base_home_orientation = R.from_quat(base_home_orientation)
        self.cable_constraint = kwargs.pop("cable_constraint", None)
        self.listening = True
        self._cable_data = None
        if self.cable_constraint is not None:
            self._cable_data = self.cable_constraint.findData("value")
            current = self._cable_data.value if self._cable_data is not None else 0.0
            if isinstance(current, (list, tuple)):
                self.joint_pos[2] = float(current[0])
            else:
                self.joint_pos[2] = float(current)

        self.pressed_keys = set()
        self.keys = ['u', 'j', 'k', 'h', 'i', 'y']
        key_joint_idx = [0, 0, 1, 1, 2, 2]
        directions = [1, -1, 1, -1, 1, -1]
        self.key_bindings = {
            k: (j, d)
            for k, j, d in zip(self.keys, key_joint_idx, directions)
        }

    def onAnimateBeginEvent(self, _event) -> None:
        if self.base_mo is None or len(self.base_mo.position.value) == 0:
            return
        dt = float(self.base_mo.getContext().dt.value)
        for key in self.pressed_keys:
            if key in self.keys:
                # later key overrides
                joint, dir = self.key_bindings[key]
                self.joint_pos[joint] += self.joint_rate[joint]*dir*dt
        np.clip(self.joint_pos, self.joint_lower_limits, self.joint_upper_limits, out=self.joint_pos)
        
        self._apply_pose()
        if self._cable_data is not None:
            self._cable_data.value = [self.joint_pos[2]]

    def onKeypressedEvent(self, event) -> None:
        key = event.get("key")
        if not isinstance(key, str):
            return
        self.pressed_keys.add(key.lower())

    def onKeyreleasedEvent(self, event) -> None:
        key = event.get("key")
        if not isinstance(key, str):
            return
        self.pressed_keys.discard(key.lower())

    def _apply_pose(self) -> None:
        rotation_delta = R.from_rotvec(self.joint_pos[1]*self.direction, degrees=True)
        base_rotation = (rotation_delta * self._base_home_orientation).as_quat()
        with self.base_mo.position.writeable() as pos:
            pos[0][0:3] = (self.base_home_position + \
                           self.direction * self.joint_pos[0]).tolist()
            pos[0][3:7] = base_rotation.tolist()
        if hasattr(self.base_mo, "rest_position") and len(self.base_mo.rest_position.value) > 0:
            with self.base_mo.rest_position.writeable() as rest:
                rest[0][0:3] = (self.base_home_position + \
                                self.direction * self.joint_pos[0]).tolist()
                rest[0][3:7] = base_rotation.tolist()

    @staticmethod
    def _normalize(vec: Sequence[float]) -> List[float]:
        arr = np.asarray(vec, dtype=float)
        norm_sq = float(np.dot(arr, arr))
        if norm_sq <= 0.0:
            raise ValueError("direction vector must be non-zero")
        return arr * (norm_sq ** -0.5)


def createScene(root: Sofa.Core.Node) -> Sofa.Core.Node:
    root.gravity = [0.0, -9.81, 0.0]
    root.dt = 0.01

    config = root.addChild("Config")
    _add_required_plugins(config)
    _add_scene_utilities(root)

    heart = _add_heart(root)
    catheter_prefab, catheter_collision, cable_constraint = _add_catheter(root)
    _attach_contact_listener(root, heart, catheter_collision)
    _attach_catheter_controller(root, catheter_prefab, cable_constraint)

    return root


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------
def _add_required_plugins(config_node: Sofa.Core.Node) -> None:
    base_plugins = [
        "Sofa.Component.AnimationLoop",
        "Sofa.Component.Collision.Detection.Algorithm",
        "Sofa.Component.Collision.Detection.Intersection",
        "Sofa.Component.Collision.Geometry",
        "Sofa.Component.Collision.Response.Contact",
        "Sofa.Component.Constraint.Lagrangian.Correction",
        "Sofa.Component.Constraint.Lagrangian.Solver",
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.Controller",
        "Sofa.Component.Engine.Select",
        "Sofa.Component.IO.Mesh",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.LinearSolver.Iterative",
        "Sofa.Component.Mass",
        "Sofa.Component.Mapping.Linear",
        "Sofa.Component.Mapping.NonLinear",
        "Sofa.Component.MechanicalLoad",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Constant",
        "Sofa.Component.Topology.Container.Dynamic",
        "Sofa.Component.Topology.Mapping",
        "Sofa.Component.Visual",
        "Sofa.GL.Component.Rendering3D",
        "Cosserat",
        "SoftRobots",
    ]

    for plugin in _unique(base_plugins + list(COSERAT_PLUGIN_LIST)):
        config_node.addObject(
            "RequiredPlugin",
            name=f"plugin_{plugin.replace('.', '_')}",
            pluginName=plugin,
            printLog=False,
        )


def _add_scene_utilities(root: Sofa.Core.Node) -> None:
    root.addObject("VisualStyle", displayFlags=DISPLAY_FLAGS)
    root.addObject("BackgroundSetting", color=BACKGROUND_COLOR)
    root.addObject("OglSceneFrame", style="Arrows", alignment="TopRight")
    root.addObject("DefaultVisualManagerLoop")
    root.addObject("FreeMotionAnimationLoop")
    root.addObject("NNCGConstraintSolver", tolerance=1e-9, maxIterations=500)
    root.addObject("CollisionPipeline")
    root.addObject("BruteForceBroadPhase")
    root.addObject("BVHNarrowPhase")
    root.addObject(
        "RuleBasedContactManager",
        name="ContactManager",
        responseParams="mu=0.15",
        response="FrictionContactConstraint",
    )
    root.addObject(
        "LocalMinDistance",
        name="Proximity",
        alarmDistance=4.0,
        contactDistance=1.5,
        angleCone=0.05,
    )


def _add_heart(root: Sofa.Core.Node) -> Sofa.Core.Node:
    if not os.path.isfile(HEART_MESH):
        raise FileNotFoundError(f"Heart mesh not found at {HEART_MESH}")

    heart = root.addChild("Heart")
    heart.addObject(
        "MechanicalObject",
        name="mstate",
        template="Rigid3d",
        position=[[*HEART_POSITION, *HEART_ORIENTATION]],
        showObject=False,
    )
    heart.addObject("UniformMass", totalMass=1.0)
    heart.addObject("FixedProjectiveConstraint", indices=[0])

    collision = heart.addChild("Collision")
    collision.addObject(
        "MeshSTLLoader",
        name="loader",
        filename=HEART_MESH,
        scale=1.0,
        triangulate=True,
    )
    collision.addObject("MeshTopology", src="@loader")
    collision.addObject("MechanicalObject")
    collision.addObject("TriangleCollisionModel", moving=False, simulated=False)
    collision.addObject("LineCollisionModel", moving=False, simulated=False)
    collision.addObject("PointCollisionModel", moving=False, simulated=False)
    collision.addObject("RigidMapping")

    visual = heart.addChild("Visual")
    visual.addObject("MeshSTLLoader", name="loader", filename=HEART_MESH)
    visual.addObject(
        "OglModel",
        src="@loader",
        color=[0.85, 0.1, 0.1, 1.0],
        updateNormals=False,
    )
    visual.addObject("RigidMapping")
    return heart


def _add_catheter(root: Sofa.Core.Node) -> Tuple[CosseratBase, Sofa.Core.Node, Sofa.Core.Object | None]:
    catheter_solver = root.addChild("CatheterSimulation")
    catheter_solver.addObject(
        "EulerImplicitSolver",
        rayleighStiffness=CATHETER_RAYLEIGH,
        rayleighMass=1e-3,
    )
    catheter_solver.addObject(
        "SparseLDLSolver",
        name="solver",
        template="CompressedRowSparseMatrixd",
    )
    catheter_solver.addObject("GenericConstraintCorrection")

    catheter_params = _build_catheter_params()
    catheter = catheter_solver.addChild(
        CosseratBase(
            parent=catheter_solver,
            params=catheter_params,
            name="catheter",
            translation=CATHETER_BASE_TRANSLATION,
            rotation=CATHETER_BASE_ROTATION,
        )
    )

    # Keep the catheter base stable when external loads (cable/contacts) are applied
    catheter.rigidBaseNode.addObject(  # type: ignore[attr-defined]
        "RestShapeSpringsForceField",
        name="BaseAttachment",
        stiffness=1e8,
        angularStiffness=1e8,
        external_points=0,
        points=0,
        template="Rigid3d",
    )

    # Visual cues on the Cosserat frame points (Dofs distributed along the catheter)
    catheter.cosseratFrame.FramesMO.showObject = True  # type: ignore[attr-defined]
    catheter.cosseratFrame.FramesMO.showObjectScale = 0.8  # type: ignore[attr-defined]

    collision = catheter.addCollisionModel()
    if hasattr(collision, "CollisionDOFs"):
        collision.CollisionDOFs.showObject = False  # type: ignore[attr-defined]

    cable_constraint = _add_actuation_cable(catheter)

    return catheter, collision, cable_constraint


def _add_actuation_cable(catheter: CosseratBase) -> Sofa.Core.Object | None:
    frame_states = np.asarray(catheter.frames3D, dtype=float)
    cable_points = _compute_cable_points(frame_states, CABLE_POINT_COUNT)
    if cable_points.shape[0] < 2:
        return None

    frame_node = catheter.rigidBaseNode.cosseratInSofaFrameNode  # type: ignore[attr-defined]
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
    attachment.addObject(
        "SkinningMapping",
        nbRef="1",
        name="CableSkinning",
    )

    cable = PullingCable(
        attachedTo=attachment,
        name="ActuationCable",
        cableGeometry=guide_positions,
        valueType="displacement",
    )

    cable_constraint = getattr(cable, "CableConstraint", None)
    cable_mo = getattr(cable, "MechanicalObject", None)
    if cable_mo is not None:
        cable_mo.showObject = True
        cable_mo.showIndices = False
        cable_mo.showObjectScale = 1.25
        cable_mo.showColor = [0.2, 0.85, 0.3, 1.0]

    return cable_constraint


def _attach_contact_listener(
    root: Sofa.Core.Node,
    heart: Sofa.Core.Node,
    catheter_collision: Sofa.Core.Node,
) -> None:
    heart_collision = heart.Collision  # type: ignore[attr-defined]
    root.addObject(
        "ContactListener",
        name="contactStats",
        collisionModel1=heart_collision.TriangleCollisionModel.getLinkPath(),
        collisionModel2=catheter_collision.PointCollisionModel.getLinkPath(),
    )


def _attach_catheter_controller(
    root: Sofa.Core.Node, catheter: CosseratBase, cable_constraint: Sofa.Core.Object | None
) -> None:
    base_mo = catheter.rigidBaseNode.RigidBaseMO  # type: ignore[attr-defined]
    base_position = list(CATHETER_BASE_POSITION)
    base_orientation = list(CATHETER_BASE_ORIENTATION)

    controller = CatheterKeyboardController(
        name="CatheterKeyboardController",
        base_mechanical_object=base_mo,
        direction=CATHETER_INSERTION_DIRECTION,
        joint_rate=[CATHETER_INSERTION_SPEED, 
                    CATHETER_ROTATION_SPEED, 
                    CABLE_PULL_INCREMENT],
        joint_upper_limits=[CATHETER_MAX_TRAVEL,
                            CATHETER_MAX_ROTATION,
                            CABLE_PULL_MAX],
        joint_lower_limits=[0.0, -CATHETER_MAX_ROTATION, CABLE_PULL_MIN],
        base_position=base_position,
        base_orientation=base_orientation,
        cable_constraint=cable_constraint,
    )
    root.addObject(controller)


def _build_catheter_params() -> Parameters:
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


def _compute_cable_points(
    frame_states: Sequence[Sequence[float]], target_count: int
    ) -> np.ndarray:
    if target_count <= 0 or len(frame_states) == 0:
        return np.empty((0, 3)) 

    frames = np.asarray(frame_states, dtype=float)
    frames = frames.reshape((-1, frames.shape[-1]))
    if frames.shape[1] >= 3:
        positions = frames[:, :3]
    else:
        positions = np.zeros((len(frames), 3))
        positions[:, : frames.shape[1]] = frames
        
    positions[:, 1] += CABLE_OFFSET
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


def _unique(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
