from typing import Iterable, List

import Sofa
import Sofa.Core

from cosserat.usefulFunctions import pluginList as _COSSERAT_PLUGIN_LIST  # type: ignore


BACKGROUND_COLOR = [0.0, 0.0, 0.0, 1.0]
DISPLAY_FLAGS = (
    "showVisualModels showBehaviorModels hideCollisionModels hideBoundingCollisionModels "
    "showForceFields hideInteractionForceFields hideWireframe showMechanicalMappings"
)

_BASE_PLUGINS = [
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


def add_required_plugins(root: Sofa.Core.Node) -> None:
    config = root.addChild("Config")
    for plugin in _unique(_BASE_PLUGINS + list(_COSSERAT_PLUGIN_LIST)):
        config.addObject(
            "RequiredPlugin",
            name=f"plugin_{plugin.replace('.', '_')}",
            pluginName=plugin,
            printLog=False,
        )


def add_scene_utilities(root: Sofa.Core.Node) -> None:
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


def _unique(values: Iterable[str]) -> List[str]:
    seen: set = set()
    ordered: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
