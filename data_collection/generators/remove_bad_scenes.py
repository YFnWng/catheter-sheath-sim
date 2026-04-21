"""Remove scenes without initial_config from a generated scenes YAML.

Usage:
    python simulation/data_collection/generators/remove_bad_scenes.py \\
        simulation/configs/generated_scenes.yaml
"""
from __future__ import annotations

import argparse
import sys

import yaml


def main():
    parser = argparse.ArgumentParser(
        description="Remove scenes without initial_config and renumber",
    )
    parser.add_argument("yaml_path", help="Path to the scenes YAML file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be removed without writing")
    args = parser.parse_args()

    with open(args.yaml_path) as f:
        data = yaml.safe_load(f) or {}

    scenes = data.get("scenes", [])
    if not scenes:
        print("No scenes found.")
        return

    kept = []
    removed = []
    for i, scene in enumerate(scenes):
        has_init = isinstance(scene, dict) and scene.get("initial_config") is not None
        if has_init:
            kept.append((i, scene))
        else:
            removed.append(i)

    print(f"Total: {len(scenes)}  kept: {len(kept)}  removed: {len(removed)}")
    if removed:
        print(f"Removing scene indices: {removed}")
    for new_idx, (old_idx, scene) in enumerate(kept):
        objects = scene.get("objects", []) if isinstance(scene, dict) else []
        if objects:
            obj = objects[0]
            name = obj.get("type", "?") if isinstance(obj, dict) else str(obj)
        else:
            name = "(free space)"
        label = f"[{old_idx}] → [{new_idx}]" if old_idx != new_idx else f"[{new_idx}]"
        print(f"  {label} {name}")

    if args.dry_run:
        print("\nDry run — no changes written.")
        return

    if not removed:
        print("Nothing to remove.")
        return

    data["scenes"] = [scene for _, scene in kept]
    with open(args.yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=None, sort_keys=False)

    print(f"\nWritten {len(kept)} scene(s) to {args.yaml_path}")


if __name__ == "__main__":
    main()
