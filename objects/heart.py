import os

import Sofa
import Sofa.Core
from scipy.spatial.transform import Rotation as R

# Resolve asset path relative to the simulation root (two levels up from this file)
_SIMULATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEART_MESH = os.path.join(
    _SIMULATION_DIR, "assets", "full_heart_clean_reduced_transseptal.stl"
)

HEART_POSITION = [60.0e-3, -75.0e-3, -20.0e-3]  # m
HEART_ORIENTATION = (
    R.from_euler("xyz", [-55.0, -20.0, 0.0], degrees=True).as_quat().tolist()
)


class HeartModel:
    """Rigid fixed heart: rigid body + STL collision + STL visual."""

    def __init__(self, root: Sofa.Core.Node) -> None:
        if not os.path.isfile(HEART_MESH):
            raise FileNotFoundError(f"Heart mesh not found at {HEART_MESH}")

        self._node = root.addChild("Heart")
        self._node.addObject(
            "MechanicalObject",
            name="mstate",
            template="Rigid3d",
            position=[[*HEART_POSITION, *HEART_ORIENTATION]],
            showObject=False,
        )
        self._node.addObject("UniformMass", totalMass=1.0, showAxisSizeFactor=0)
        self._node.addObject("FixedProjectiveConstraint", indices=[0])

        collision = self._node.addChild("Collision")
        collision.addObject(
            "MeshSTLLoader",
            name="loader",
            filename=HEART_MESH,
            scale=1.0e-3,  # STL mesh is in mm; convert to m
            triangulate=True,
        )
        collision.addObject("MeshTopology", src="@loader")
        collision.addObject("MechanicalObject", showObject=False)
        collision.addObject("TriangleCollisionModel", moving=False, simulated=False)
        collision.addObject("LineCollisionModel", moving=False, simulated=False)
        collision.addObject("PointCollisionModel", moving=False, simulated=False)
        collision.addObject("RigidMapping")

        visual = self._node.addChild("Visual")
        visual.addObject("MeshSTLLoader", name="loader", filename=HEART_MESH, scale=1.0e-3)
        visual.addObject(
            "OglModel",
            src="@loader",
            color=[0.85, 0.1, 0.1, 0.5],
            updateNormals=False,
        )
        visual.addObject("RigidMapping")

    @property
    def triangle_collision_model_path(self) -> str:
        return self._node.Collision.TriangleCollisionModel.getLinkPath()
