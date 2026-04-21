# Simulation of tendon-driven continuum robots

SOFA-based simulation of tendon-driven Cosserat rod catheters with environmental contacts, automated data collection, and real-time diagnostics.

## Demo

https://github.com/user-attachments/assets/a82aca67-b4e6-4476-8078-0bbbcf658664

## Installation

Tested with:

- SOFA v25.12 binary with Python 3.10
- SofaPython3
- Cosserat plugin (out-of-tree build)
- Sofa.Qt (for GUI scenes)
- PyQt5 + pyqtgraph (for diagnostic plotter)
- h5py (for trajectory recording)

## Directory structure

```
simulation/
  configs/
    catheter_ablation.yaml          # Rod geometry, physics, actuation limits
    catheter_data_collection.yaml   # Scenes + generator config for batch collection
  robots/
    catheter.py                     # CatheterRobot: Cosserat rod + cables + collision
  objects/
    fixed_rigid_body.py             # FixedRigidBody + registered models (Heart, Pipeline, Ring, ...)
  controllers/
    keyboard_controller.py          # Interactive keyboard control (insertion, rotation, cable)
    plot_controller.py              # Reads SOFA state each step → diagnostic plotter
  scenes/
    catheter_heart.py               # Interactive scene with keyboard control + plotter
    collect_data.py                 # Automated data collection (GUI or headless)
    replay_data.py                  # Replay recorded HDF5 trajectories
  data_collection/
    generators/
      base.py                       # InputGenerator ABC
      sweep.py                      # Grid sweep through joint space
      sinusoidal.py                 # Incommensurate-frequency sinusoidal excitation
    collector.py                    # DataCollectorController (records to HDF5)
    schema.py                       # TrajectoryRecord + HDF5 read/write
    data/                           # Collected .h5 files (gitignored)
  utils/
    scene.py                        # SOFA plugin loading, scene utilities
    plotter.py                      # DiagnosticPlotter (subprocess pyqtgraph window)
    sofa_reader.py                  # SofaReader: extracts rod state from SOFA
    sofa_data.py                    # SofaGroundTruth dataclass
    sofa_env.py                     # SOFA Python path setup for headless mode
    message_handler.py              # SOFA message handler for headless runs
    cable_utils.py                  # Cable routing geometry
  assets/                           # STL mesh files
```

## Scenes

### Interactive (`catheter_heart.py`)

```bash
runSofa -g qt simulation/scenes/catheter_heart.py
```

Keyboard control: `u/j` insertion, `k/h` rotation, `i/y` cable tension. Four-panel diagnostic plotter shows base translation, rotation, tendon force, and per-node contact force in real time.

### Data collection (`collect_data.py`)

Collects trajectory data from multiple environment configurations defined in `catheter_data_collection.yaml`.

**GUI mode** (one scene, with plotter):
```bash
runSofa -g qt simulation/scenes/collect_data.py
```

**Headless mode** (all scenes, no GUI — faster):
```bash
python simulation/scenes/collect_data.py
```

Environment variables: `COLLECT_GENERATOR` (`sweep` | `sinusoidal`), `COLLECT_WARMUP`, `COLLECT_MAX_STEPS`, `COLLECT_SCENE_IDX`.

### Replay (`replay_data.py`)

Replays recorded HDF5 trajectories with the original environment reconstructed from metadata.

```bash
REPLAY_FILE=sweep_PipelineModel_RingModel_20260417.h5 \
    runSofa -g qt simulation/scenes/replay_data.py
```

`REPLAY_FILE` is resolved relative to `data_collection/data/` if not absolute.

## Environment configuration

Scenes are defined as lists of environment objects in YAML. Each object is a registered model name or a dict with parameter overrides:

```yaml
scenes:
  - - {type: PipelineModel, position: [0, 0, -0.02], scale: 0.5}
    - {type: RingModel, scale: 0.01}
  - - {type: HeartInsideModel}
```

Available models: `HeartModel`, `HeartInsideModel`, `TurbineModel`, `PipelineModel`, `RingModel`. Custom meshes can be added via `FixedRigidBody` with `mesh_path`.

## Robot

`CatheterRobot` is a tendon-driven Cosserat rod with configurable:
- Rod geometry (length, sections, radius)
- Variable stiffness per section (`stiffness_sections` in YAML)
- Multiple cable actuators
- Collision model for contact detection

Joint interface: `[insertion (m), rotation (deg), cable_0 (N), ...]` with `joint_names` and `joint_types` properties for downstream use.

## HDF5 trajectory format

Each `.h5` file contains per-timestep arrays:

| Dataset | Shape | Description |
|---------|-------|-------------|
| `timestamps` | (N,) | Simulation time (s) |
| `frame_poses` | (N, n_frames, 7) | Rigid3d poses [x,y,z,qx,qy,qz,qw] |
| `strain_coords` | (N, n_sections, 3) | Curvature per section |
| `joint_commands` | (N, n_joints) | Control inputs |
| `contact_force_body` | (N, n_frames, 3) | Per-node contact force (body frame) |

Metadata attributes (JSON): `scene_objects`, `robot_type`, `cable_mode`, `dt`, `generator`.
