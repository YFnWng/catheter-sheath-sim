"""Fixed rigid body from an STL mesh — collision + visual.

Includes a model registry for YAML-driven scene configuration::

    from simulation.objects.fixed_rigid_body import add_environment

    for obj in add_environment(root, cfg["environment"][scene_idx]):
        pass  # each obj is a FixedRigidBody with collision
"""
from __future__ import annotations

import os
from typing import Sequence

import Sofa
import Sofa.Core
from scipy.spatial.transform import Rotation as R

_SIMULATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FixedRigidBody:
    """Add any STL mesh to a SOFA scene as a fixed rigid body with collision.

    Parameters
    ----------
    root : Sofa.Core.Node
        Parent SOFA node.
    mesh_path : str
        Absolute path to the STL file.
    name : str
        Name of the child node created under *root*.
    position : (3,) sequence
        World-frame position [x, y, z] in metres.
    orientation_euler_xyz_deg : (3,) sequence
        Orientation as XYZ intrinsic Euler angles in degrees.
    scale : float
        Uniform scale applied to the mesh at load time (e.g. 1e-3 for mm→m).
    color : (4,) sequence
        RGBA colour for the visual model.
    mass : float
        Total mass (only affects inertia display; body is fixed).
    """

    def __init__(
        self,
        root: Sofa.Core.Node,
        mesh_path: str,
        name: str = "RigidBody",
        position: Sequence[float] = (0.0, 0.0, 0.0),
        orientation_euler_xyz_deg: Sequence[float] = (0.0, 0.0, 0.0),
        scale: float = 1.0,
        color: Sequence[float] = (0.8, 0.8, 0.8, 1.0),
        mass: float = 1.0,
    ) -> None:
        if not os.path.isfile(mesh_path):
            raise FileNotFoundError(f"Mesh not found at {mesh_path}")

        orientation = R.from_euler(
            "xyz", list(orientation_euler_xyz_deg), degrees=True,
        ).as_quat().tolist()

        self._node = root.addChild(name)
        self._node.addObject(
            "MechanicalObject",
            name="mstate",
            template="Rigid3d",
            position=[[*position, *orientation]],
            showObject=False,
        )
        self._node.addObject("UniformMass", totalMass=mass, showAxisSizeFactor=0)
        self._node.addObject("FixedProjectiveConstraint", indices=[0])

        collision = self._node.addChild("Collision")
        collision.addObject(
            "MeshSTLLoader",
            name="loader",
            filename=mesh_path,
            scale=scale,
            triangulate=True,
        )
        collision.addObject("MeshTopology", src="@loader")
        collision.addObject("MechanicalObject", showObject=False)
        collision.addObject("TriangleCollisionModel", moving=False, simulated=False)
        collision.addObject("LineCollisionModel", moving=False, simulated=False)
        collision.addObject("PointCollisionModel", moving=False, simulated=False)
        collision.addObject("RigidMapping")

        visual = self._node.addChild("Visual")
        visual.addObject(
            "MeshSTLLoader", name="loader", filename=mesh_path, scale=scale,
        )
        visual.addObject(
            "OglModel",
            src="@loader",
            color=list(color),
            updateNormals=False,
        )
        visual.addObject("RigidMapping")

    @property
    def triangle_collision_model_path(self) -> str:
        return self._node.Collision.TriangleCollisionModel.getLinkPath()

    @property
    def point_collision_model_path(self) -> str:
        return self._node.Collision.PointCollisionModel.getLinkPath()


# ── Convenience subclasses ──────────────────────────────────────────────

HEART_MESH = os.path.join(
    _SIMULATION_DIR, "assets", "full_heart_clean_reduced_transseptal.stl",
)

HEART_INSIDE_MESH = os.path.join(
    _SIMULATION_DIR, "assets", "heart_inside.stl",
)

TURBINE_MESH = os.path.join(
    _SIMULATION_DIR, "assets", "turbine_view.stl",
)

PIPELINES_MESH = os.path.join(
    _SIMULATION_DIR, "assets", "pipelines_new_coarse.stl",
)

RING_MESH = os.path.join(
    _SIMULATION_DIR, "assets", "ring.stl",
)




class HeartModel(FixedRigidBody):
    """Pre-configured full heart mesh."""

    _DEFAULTS = dict(
        mesh_path=HEART_MESH, name="Heart",
        position=[60.0e-3, -75.0e-3, -20.0e-3],
        orientation_euler_xyz_deg=[-55.0, -20.0, 0.0],
        scale=1.0e-3, color=[0.85, 0.1, 0.1, 0.5],
    )

    def __init__(self, root: Sofa.Core.Node, **overrides) -> None:
        kw = {**self._DEFAULTS, **overrides}
        super().__init__(root, **kw)


class HeartInsideModel(FixedRigidBody):
    """Pre-configured inside surface mesh of the amazon heart model."""

    _DEFAULTS = dict(
        mesh_path=HEART_INSIDE_MESH, name="HeartInside",
        position=[0.0, 0.0, 0.0],
        orientation_euler_xyz_deg=[0.0, 0.0, 0.0],
        scale=1.0e0, color=[0.85, 0.1, 0.1, 0.5],
    )

    def __init__(self, root: Sofa.Core.Node, **overrides) -> None:
        kw = {**self._DEFAULTS, **overrides}
        super().__init__(root, **kw)


class TurbineModel(FixedRigidBody):
    """Pre-configured turbine mesh."""

    _DEFAULTS = dict(
        mesh_path=TURBINE_MESH, name="Turbine",
        position=[0.0, -40.0e-3, -20.0e-3],
        orientation_euler_xyz_deg=[0.0, 0.0, 0.0],
        scale=1.0e0, color=[0.7, 0.7, 0.7, 1.0],
    )

    def __init__(self, root: Sofa.Core.Node, **overrides) -> None:
        kw = {**self._DEFAULTS, **overrides}
        super().__init__(root, **kw)


class PipelineModel(FixedRigidBody):
    """Pre-configured pipeline mesh."""

    _DEFAULTS = dict(
        mesh_path=PIPELINES_MESH, name="Pipelines",
        position=[0.0, 0.0, -20.0e-3],
        orientation_euler_xyz_deg=[0.0, 0.0, 0.0],
        scale=5.0e-1, color=[0.7, 0.7, 0.7, 1.0],
    )

    def __init__(self, root: Sofa.Core.Node, **overrides) -> None:
        kw = {**self._DEFAULTS, **overrides}
        super().__init__(root, **kw)


class RingModel(FixedRigidBody):
    """Pre-configured ring mesh."""

    _DEFAULTS = dict(
        mesh_path=RING_MESH, name="Ring",
        position=[0.0, 0.0, 0.0],
        orientation_euler_xyz_deg=[0.0, 0.0, 0.0],
        scale=1.0e-2, color=[0.7, 0.7, 0.7, 1.0],
    )

    def __init__(self, root: Sofa.Core.Node, **overrides) -> None:
        kw = {**self._DEFAULTS, **overrides}
        super().__init__(root, **kw)


# ── Model registry + factory ──────────────────────────────────────────────

MODEL_REGISTRY = {
    "HeartModel": HeartModel,
    "HeartInsideModel": HeartInsideModel,
    "TurbineModel": TurbineModel,
    "PipelineModel": PipelineModel,
    "RingModel": RingModel,
}


def add_environment(root, scene_objects):
    """Instantiate environment objects from a YAML scene list.

    Parameters
    ----------
    root : Sofa.Core.Node
        Parent SOFA node.
    scene_objects : list of str or list of dict
        Each entry is either a model name (str) looked up in
        ``MODEL_REGISTRY``, or a dict ``{"type": "...", ...}``
        where extra keys are passed to ``FixedRigidBody.__init__``.

    Returns
    -------
    list of FixedRigidBody
        The instantiated environment objects.

    Example YAML::

        scenes:
          - [PipelineModel, RingModel]
          - [HeartModel]
          - [{type: FixedRigidBody, mesh_path: assets/custom.stl,
              name: Custom, scale: 0.001}]
    """
    objects = []
    for entry in scene_objects:
        if isinstance(entry, str):
            cls = MODEL_REGISTRY.get(entry)
            if cls is None:
                raise ValueError(
                    f"Unknown model: {entry!r}. "
                    f"Available: {list(MODEL_REGISTRY.keys())}"
                )
            objects.append(cls(root))
        elif isinstance(entry, dict):
            entry = dict(entry)
            type_name = entry.pop("type", "FixedRigidBody")
            if type_name == "FixedRigidBody":
                objects.append(FixedRigidBody(root, **entry))
            else:
                cls = MODEL_REGISTRY.get(type_name)
                if cls is None:
                    raise ValueError(f"Unknown model: {type_name!r}")
                # Pass remaining keys as overrides to the registered model
                objects.append(cls(root, **entry))
        else:
            raise ValueError(f"Invalid scene entry: {entry!r}")
    return objects