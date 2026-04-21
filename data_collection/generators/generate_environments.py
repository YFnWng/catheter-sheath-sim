"""Generate random single-obstacle environments for data collection.

Each environment contains one obstacle — either a RingModel or a SlabModel —
with randomly sampled position, orientation, and scale.  Generated scenes are
appended to a dedicated YAML file that ``collect_data.py`` can consume.

Usage:
    python -m simulation.data_collection.generators.generate_environments \\
        --n 20 \\
        --output simulation/configs/generated_scenes.yaml

    # Then manually set initial configs:
    python simulation/scenes/collect_data.py --init \\
        --scenes simulation/configs/generated_scenes.yaml

    # Then collect data headless:
    COLLECT_SCENES=simulation/configs/generated_scenes.yaml \\
        python simulation/scenes/collect_data.py
"""
from __future__ import annotations

import argparse
import os
import random
from typing import Any, Dict, List

import yaml


# ── Sampling ranges per model type ────────────────────────────────────────
# Position in metres, orientation in degrees, scale dimensionless.
# The catheter rod occupies roughly z ∈ [-0.16, 0] with tip near origin.
# Objects are placed in the reachable workspace around the distal half.

OBSTACLE_SPECS: Dict[str, Dict[str, Any]] = {
    "RingModel": {
        "position": {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.00, 0.04),
        },
        "orientation_euler_xyz_deg": {
            "x": (-45.0, 45.0),
            "y": (-45.0, 45.0),
            "z": (-45.0, 45.0),
        },
        "scale": (0.005, 0.025),
    },
    "SlabModel": {
        "position": {
            "x": (-0.1, 0.1),
            "y": (-0.1, 0.1),
            "z": (-0.04, 0.04),
        },
        "orientation_euler_xyz_deg": {
            "x": (-90.0, 90.0),
            "y": (-90.0, 90.0),
            "z": (-30.0, 30.0),
        },
        "scale": (0.05, 0.2),
    },
}

OBSTACLE_TYPES = list(OBSTACLE_SPECS.keys())


def _sample_obstacle(rng: random.Random) -> Dict[str, Any]:
    """Sample one random obstacle entry (dict format for YAML scenes)."""
    obs_type = rng.choice(OBSTACLE_TYPES)
    spec = OBSTACLE_SPECS[obs_type]

    pos_ranges = spec["position"]
    position = [
        round(rng.uniform(*pos_ranges["x"]), 5),
        round(rng.uniform(*pos_ranges["y"]), 5),
        round(rng.uniform(*pos_ranges["z"]), 5),
    ]

    ori_ranges = spec["orientation_euler_xyz_deg"]
    orientation = [
        round(rng.uniform(*ori_ranges["x"]), 1),
        round(rng.uniform(*ori_ranges["y"]), 1),
        round(rng.uniform(*ori_ranges["z"]), 1),
    ]

    scale = round(rng.uniform(*spec["scale"]), 5)

    entry: Dict[str, Any] = {
        "type": obs_type,
        "position": position,
        "orientation_euler_xyz_deg": orientation,
        "scale": scale,
    }

    return entry


def generate_environments(
    n: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate *n* single-obstacle environments plus one free-space scene.

    The first scene is always free-space (no objects).  The remaining *n*
    scenes each contain one random obstacle.

    Returns a list of scene dicts, each with ``objects`` list and no
    ``initial_config`` (to be set later via manual init mode).
    """
    rng = random.Random(seed)
    scenes: List[Dict[str, Any]] = [{"objects": []}]  # free-space first
    for _ in range(n):
        scenes.append({"objects": [_sample_obstacle(rng)]})
    return scenes


def main():
    parser = argparse.ArgumentParser(
        description="Generate random single-obstacle environments for data collection",
    )
    parser.add_argument(
        "--n", type=int, default=20,
        help="Number of environments to generate (default: 20)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output", type=str,
        default="simulation/configs/generated_scenes.yaml",
        help="Output YAML file path (scenes are appended if file exists)",
    )
    args = parser.parse_args()

    new_scenes = generate_environments(args.n, seed=args.seed)

    # Load existing file if present
    output_path = args.output
    if os.path.isfile(output_path):
        with open(output_path) as f:
            existing = yaml.safe_load(f) or {}
        existing_scenes = existing.get("scenes", [])
        print(f"Loaded {len(existing_scenes)} existing scene(s) from {output_path}")
    else:
        existing_scenes = []
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    all_scenes = existing_scenes + new_scenes

    with open(output_path, "w") as f:
        yaml.dump(
            {"scenes": all_scenes},
            f,
            default_flow_style=None,
            sort_keys=False,
            allow_unicode=True,
        )

    print(f"Appended {len(new_scenes)} scene(s) → {len(all_scenes)} total in {output_path}")
    for i, scene in enumerate(new_scenes):
        idx = len(existing_scenes) + i
        objects = scene["objects"]
        has_init = "initial_config" in scene
        status = "ready" if has_init else "needs init"
        if not objects:
            print(f"  [{idx}] {'(free space)':<12}  ({status})")
        else:
            obj = objects[0]
            pos_str = "[" + ", ".join(f"{v: .4f}" for v in obj["position"]) + "]"
            print(f"  [{idx}] {obj['type']:<12} pos={pos_str}  "
                  f"scale={obj['scale']:.4f}  ({status})")


if __name__ == "__main__":
    main()
