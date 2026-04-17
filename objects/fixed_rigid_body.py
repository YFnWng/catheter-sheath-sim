"""Fixed rigid body from an STL mesh — collision + visual."""
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



class HeartModel(FixedRigidBody):
    """Pre-configured full heart mesh."""

    def __init__(self, root: Sofa.Core.Node) -> None:
        super().__init__(
            root,
            mesh_path=HEART_MESH,
            name="Heart",
            position=[60.0e-3, -75.0e-3, -20.0e-3],
            orientation_euler_xyz_deg=[-55.0, -20.0, 0.0],
            scale=1.0e-3,
            color=[0.85, 0.1, 0.1, 0.5],
        )


class HeartInsideModel(FixedRigidBody):
    """Pre-configured inside surface mesh of the amazon heart model (a different heart)."""

    def __init__(self, root: Sofa.Core.Node) -> None:
        super().__init__(
            root,
            mesh_path=HEART_INSIDE_MESH,
            name="HeartInside",
            position=[0.0, 0.0, 0.0],
            orientation_euler_xyz_deg=[0.0, 0.0, 0.0],
            scale=1.0e0,
            color=[0.85, 0.1, 0.1, 0.5],
        )


class TurbineModel(FixedRigidBody):
    """Pre-configured turbine mesh."""

    def __init__(self, root: Sofa.Core.Node) -> None:
        super().__init__(
            root,
            mesh_path=TURBINE_MESH,
            name="Turbine",
            position=[0.0, 0.0, 0.0],
            orientation_euler_xyz_deg=[0.0, 0.0, 0.0],
            scale=1.0e0,
            color=[0.7, 0.7, 0.7, 1.0],
        )


class PipelineModel(FixedRigidBody):
    """Pre-configured pipeline mesh."""

    def __init__(self, root: Sofa.Core.Node) -> None:
        super().__init__(
            root,
            mesh_path=PIPELINES_MESH,
            name="Pipelines",
            position=[0.0, 0.0, 0.0],
            orientation_euler_xyz_deg=[0.0, 0.0, 0.0],
            scale=1.0e0,
            color=[0.7, 0.7, 0.7, 1.0],
        )