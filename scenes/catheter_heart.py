import math
import os
from typing import Iterable, List, Sequence, Tuple

from scipy.spatial.transform import Rotation as R

import Sofa
import Sofa.Core

SCENE_DIR = os.path.dirname(os.path.abspath(__file__))
HEART_MESH = os.path.join(SCENE_DIR, "assets/full_heart_clean_reduced.stl")

from cosserat.CosseratBase import CosseratBase  # type: ignore  # pylint: disable=import-error
from cosserat.usefulFunctions import pluginList as COSERAT_PLUGIN_LIST  # type: ignore
from useful.params import (  # type: ignore  # pylint: disable=import-error
    BeamGeometryParameters,
    BeamPhysicsParametersNoInertia,
    Parameters,
)


# ---------------------------------------------------------------------------
# Scene-wide parameters
# ---------------------------------------------------------------------------
BACKGROUND_COLOR = [0.05, 0.05, 0.08, 1.0]
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
CATHETER_RADIUS = 1.5  # mm
CATHETER_MASS = 0.04  # kg
CATHETER_YOUNG_MODULUS = 2.5e6  # Pa
CATHETER_POISSON = 0.38
CATHETER_RAYLEIGH = 0.05
CATHETER_BASE_TRANSLATION = [0.0, 0.0, -CATHETER_LENGTH]
CATHETER_BASE_ROTATION = [0.0, -90.0, 0.0]
CATHETER_INSERTION_SPEED = 30.0  # mm per second along +Z
CATHETER_MAX_TRAVEL = 80.0  # mm
CATHETER_BASE_ORIENTATION = (
    R.from_euler("xyz", CATHETER_BASE_ROTATION, degrees=True).as_quat().tolist()
)
# CATHETER_INSERTION_DIRECTION = R.from_euler(
#     "xyz", CATHETER_BASE_ROTATION, degrees=True
# ).apply([1.0, 0.0, 0.0]).tolist()
CATHETER_INSERTION_DIRECTION = [0.0, 0.0, 1.0]


class CatheterInsertionController(Sofa.Core.Controller):
    """Simple scripted insertion that advances the base along a direction."""

    def __init__(self, *args, **kwargs) -> None:
        base_mo = kwargs.pop("base_mechanical_object", None)
        direction = kwargs.pop("direction", [0.0, 0.0, 1.0])
        speed = kwargs.pop("speed", 0.0)
        max_distance = kwargs.pop("max_distance", 0.0)
        base_translation = kwargs.pop("base_translation", [0.0, 0.0, 0.0])
        base_orientation = kwargs.pop("base_orientation", [0.0, 0.0, 0.0, 1.0])
        super().__init__(*args, **kwargs)
        self.listening = True
        self.base_mo = base_mo
        self.direction = self._normalize(direction)
        self.speed = max(speed, 0.0)
        self.max_distance = max(max_distance, 0.0)
        self.travelled = 0.0
        self.start_pos = list(base_translation)
        self._orientation = list(base_orientation)
        self._command = 0

    def onAnimateBeginEvent(self, _event) -> None:
        if self.speed > 0.0 and self._command != 0:
            context = self.base_mo.getContext()
            dt = float(context.dt.value)
            proposed = self.travelled + self.speed * dt * self._command
            self.travelled = min(max(proposed, 0.0), self.max_distance)
        self._apply_base_translation()

    def onKeypressedEvent(self, event) -> None:
        print(f"Key pressed: {event['key']}")
        key = event["key"]
        if isinstance(key, str):
            key = key.lower()
            if key == "i":
                self._command = 1
            elif key == "k":
                self._command = -1

    def onKeyreleasedEvent(self, event) -> None:
        key = event["key"]
        if isinstance(key, str) and key.lower() in ("i", "k"):
            self._command = 0

    @staticmethod
    def _normalize(vec: Sequence[float]) -> List[float]:
        norm = math.sqrt(sum(v * v for v in vec))
        return [0.0, 0.0, 0.0] if norm == 0.0 else [v / norm for v in vec]

    def _apply_base_translation(self) -> None:
        if self.base_mo is None:
            return
        if len(self.base_mo.position.value) == 0:
            return
        with self.base_mo.position.writeable() as pos:
            for i in range(3):
                pos[0][i] = self.start_pos[i] + self.direction[i] * self.travelled
            for i in range(4):
                pos[0][3 + i] = self._orientation[i]
        if hasattr(self.base_mo, "rest_position") and len(self.base_mo.rest_position.value) > 0:
            with self.base_mo.rest_position.writeable() as rest:
                for i in range(3):
                    rest[0][i] = self.start_pos[i] + self.direction[i] * self.travelled
                for i in range(4):
                    rest[0][3 + i] = self._orientation[i]


def createScene(root: Sofa.Core.Node) -> Sofa.Core.Node:
    root.gravity = [0.0, -9.81, 0.0]
    root.dt = 0.01

    config = root.addChild("Config")
    _add_required_plugins(config)
    _add_scene_utilities(root)

    heart = _add_heart(root)
    catheter_prefab, catheter_collision = _add_catheter(root)
    _attach_contact_listener(root, heart, catheter_collision)
    _attach_insertion_controller(root, catheter_prefab)

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
        color=[0.85, 0.1, 0.1, 0.5],
        updateNormals=False,
    )
    visual.addObject("RigidMapping")
    return heart


def _add_catheter(root: Sofa.Core.Node) -> Tuple[CosseratBase, Sofa.Core.Node]:
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

    # Visual cues on the Cosserat frame points (Dofs distributed along the catheter)
    catheter.cosseratFrame.FramesMO.showObject = True  # type: ignore[attr-defined]
    catheter.cosseratFrame.FramesMO.showObjectScale = 0.8  # type: ignore[attr-defined]

    collision = catheter.addCollisionModel()
    if hasattr(collision, "CollisionDOFs"):
        collision.CollisionDOFs.showObject = False  # type: ignore[attr-defined]

    return catheter, collision

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


def _attach_insertion_controller(root: Sofa.Core.Node, catheter: CosseratBase) -> None:
    base_mo = catheter.rigidBaseNode.RigidBaseMO  # type: ignore[attr-defined]
    base_translation = list(CATHETER_BASE_TRANSLATION)
    base_orientation = list(CATHETER_BASE_ORIENTATION)

    controller = CatheterInsertionController(
        name="CatheterKeyboardController",
        base_mechanical_object=base_mo,
        direction=CATHETER_INSERTION_DIRECTION,
        speed=CATHETER_INSERTION_SPEED,
        max_distance=CATHETER_MAX_TRAVEL,
        base_translation=base_translation,
        base_orientation=base_orientation,
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


def _unique(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
