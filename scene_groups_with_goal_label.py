#!/usr/bin/env python3
"""
Build scene_groups_with_goal_label.json from agument.json.

Usage:
  python3 scene_groups_with_goal_label.py

Requires:
  - agument.json in the same directory

Output:
  - scene_groups_with_goal_label.json (for qwen_image.py and json_parsing.py)

Workflow:
  1. Run this script -> scene_groups_with_goal_label.json
  2. Run qwen_image.py -> qwen2image_v2/*.png
  3. Run postprocessing_agument.py -> 6 augmented JSON files
"""

from pathlib import Path
import json
from collections import defaultdict


def build_groups_from_agument(agument):
    """Group agument entries by identical scene (floorplan, objects, people, task)."""
    scene_to_entries = defaultdict(list)
    for key, entry in agument.items():
        scene = entry.get("scene", {})
        fp = tuple(sorted(scene.get("floorplan", [])))
        obj = tuple(sorted(scene.get("objects", [])))
        people = tuple(sorted(scene.get("people", [])))
        task = entry.get("task", "")
        key_scene = (fp, obj, people, task)
        scene_to_entries[key_scene].append((int(key), entry))

    groups = []
    for group_id, (key_scene, entries) in enumerate(
        sorted(scene_to_entries.items(), key=lambda x: min(e[0] for e in x[1])), start=1
    ):
        fp, obj, people, task = key_scene
        entries.sort(key=lambda x: x[0])
        scene_indices = [e[0] for e in entries]
        goal_label = [
            {"goal": e[1]["goal"], "label": e[1]["label"], "scene_index": e[0]}
            for e in entries
        ]
        groups.append({
            "group_id": group_id,
            "count": len(entries),
            "scene_indices": scene_indices,
            "floorplan": list(fp),
            "objects": list(obj),
            "people": list(people),
            "task": task,
            "goal_label": goal_label,
        })
    return groups


def main():
    base_dir = Path(__file__).resolve().parent
    agument_path = base_dir / "agument.json"
    out_path = base_dir / "scene_groups_with_goal_label.json"

    if not agument_path.exists():
        raise FileNotFoundError(f"Required: {agument_path.name}")

    with agument_path.open("r", encoding="utf-8") as f:
        agument = json.load(f)

    groups = build_groups_from_agument(agument)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(groups, f, ensure_ascii=False, indent=2)

    print(f"Saved {out_path} ({len(groups)} groups)")


if __name__ == "__main__":
    main()
