#!/usr/bin/env python3
"""
Generate MLIP augmented datasets with image metadata and curated splits.

Steps performed:
1. Load the original `agument.json` plus the precomputed
   `scene_groups_with_goal_label_with_img.json` that maps scene indices to
   `group_id` and synthesized image path.
2. Remove duplicate instructions per `(group_id, goal)` pair while injecting
   the `group_id` / `img_path` fields into each scene entry. Writes
   `agument_with_img_unique.json`.
3. Split the unique set into calibration / validation partitions using a
   fixed `group_id` whitelist. Writes
   `agument_with_img_unique_cal.json` and `agument_with_img_unique_val.json`.
4. Build the 180-sample subset with reproducible sampling rules and split it
   into calibration / validation partitions. Writes:
     - `agument_with_img_unique_subset_180.json`
     - `agument_with_img_unique_subset_180_cal.json`
     - `agument_with_img_unique_subset_180_val.json`
"""

import json
import random
from collections import OrderedDict, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List

# Fixed configuration derived from the existing dataset creation process.
CALIBRATION_GROUP_IDS = {1, 2, 3, 19, 20, 21, 34, 35, 36}
SPECIAL_GROUP_IDS = set(range(37, 49))  # Treated differently for subset_180 sampling
RANDOM_SEED = 42


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(data, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def build_scene_lookup(groups: Iterable[dict]) -> Dict[int, dict]:
    """Create a mapping from scene index to group/image metadata."""
    lookup: Dict[int, dict] = {}
    duplicates = []
    for group in groups:
        group_id = group["group_id"]
        img_path = group.get("img_path")
        goal_entries: List[dict] = group.get("goal_label", [])
        scene_indices: List[int] = group.get("scene_indices", [])

        for idx, scene_index in enumerate(scene_indices):
            scene_idx_int = int(scene_index)
            if scene_idx_int in lookup:
                duplicates.append(scene_idx_int)
                continue

            entry = {
                "group_id": group_id,
                "img_path": img_path,
            }
            if idx < len(goal_entries):
                entry["goal"] = goal_entries[idx].get("goal")
                entry["label"] = goal_entries[idx].get("label")
            lookup[scene_idx_int] = entry

    if duplicates:
        duplicate_preview = ", ".join(map(str, duplicates[:10]))
        raise ValueError(
            "Duplicate scene indices detected while building lookup table: "
            f"{duplicate_preview} (total {len(duplicates)})"
        )
    return lookup


def create_unique_dataset(base_data, index_lookup):
    """Remove duplicates and inject image metadata."""
    unique = OrderedDict()
    seen_pairs = set()
    dropped = 0

    for key in sorted(base_data, key=lambda x: int(x)):
        source_entry = base_data[key]
        scene_idx = int(key)
        meta = index_lookup.get(scene_idx)
        if meta is None:
            # No mapping implies the entry cannot be linked to an image.
            dropped += 1
            continue

        dedup_key = (meta["group_id"], source_entry["goal"])
        if dedup_key in seen_pairs:
            dropped += 1
            continue

        # Ensure label consistency when we have goal metadata.
        if "label" in meta and meta["label"] is not None:
            assert (
                source_entry["label"] == meta["label"]
            ), "Label mismatch between base data and group metadata"

        seen_pairs.add(dedup_key)
        new_entry = deepcopy(source_entry)
        new_scene = deepcopy(source_entry.get("scene", {}))
        new_scene["group_id"] = meta["group_id"]
        new_scene["img_path"] = meta["img_path"]
        new_entry["scene"] = new_scene
        unique[key] = new_entry

    return unique, dropped


def split_calibration_validation(dataset, calibration_group_ids: Iterable[int]):
    calibration_ids = set(calibration_group_ids)
    cal_set = OrderedDict()
    val_set = OrderedDict()

    for key in sorted(dataset, key=lambda x: int(x)):
        entry = dataset[key]
        target = cal_set if entry["scene"]["group_id"] in calibration_ids else val_set
        target[key] = deepcopy(entry)

    return cal_set, val_set


def create_subset_180(dataset, rng: random.Random):
    grouped = defaultdict(list)
    for key, entry in dataset.items():
        grouped[entry["scene"]["group_id"]].append((key, entry))

    selected: Dict[str, dict] = {}

    for group_id in sorted(grouped):
        entries = grouped[group_id]
        if not entries:
            continue

        if group_id in SPECIAL_GROUP_IDS:
            count = min(2, len(entries))
            for key, entry in rng.sample(entries, count):
                selected[key] = deepcopy(entry)
            continue

        per_label = defaultdict(list)
        for key, entry in entries:
            per_label[entry["label"]].append((key, entry))

        for label in (0, 1, 2):
            samples = per_label.get(label)
            if not samples:
                continue
            take = min(2, len(samples))
            for key, entry in rng.sample(samples, take):
                selected[key] = deepcopy(entry)

    return OrderedDict(
        sorted(selected.items(), key=lambda item: int(item[0]))
    )


def split_subset_cal_val(subset, cal_ratio: float, rng: random.Random):
    grouped = defaultdict(list)
    for key, entry in subset.items():
        grouped[(entry["task"], entry["label"])].append(key)

    cal_keys: List[str] = []
    val_keys: List[str] = []

    for group_key in sorted(grouped):
        keys = grouped[group_key][:]
        keys.sort(key=lambda x: int(x))
        rng.shuffle(keys)

        n_total = len(keys)
        n_cal = int(n_total * cal_ratio)
        cal_keys.extend(keys[:n_cal])
        val_keys.extend(keys[n_cal:])

    cal = OrderedDict(
        (key, deepcopy(subset[key])) for key in sorted(cal_keys, key=lambda x: int(x))
    )
    val = OrderedDict(
        (key, deepcopy(subset[key])) for key in sorted(val_keys, key=lambda x: int(x))
    )
    return cal, val


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    print(f"Working directory: {base_dir}")

    base_data = load_json(base_dir / "agument.json")
    groups = load_json(base_dir / "scene_groups_with_goal_label_with_img.json")

    scene_lookup = build_scene_lookup(groups)
    unique_dataset, dropped = create_unique_dataset(base_data, scene_lookup)
    dump_json(unique_dataset, base_dir / "agument_with_img_unique.json")
    print(f"Unique dataset: {len(unique_dataset)} entries (dropped {dropped})")

    cal_set, val_set = split_calibration_validation(unique_dataset, CALIBRATION_GROUP_IDS)
    dump_json(cal_set, base_dir / "agument_with_img_unique_cal.json")
    dump_json(val_set, base_dir / "agument_with_img_unique_val.json")
    print(f"Calibration set: {len(cal_set)} entries")
    print(f"Validation set: {len(val_set)} entries")

    rng_subset = random.Random(RANDOM_SEED)
    subset_180 = create_subset_180(val_set, rng_subset)
    dump_json(subset_180, base_dir / "agument_with_img_unique_subset_180.json")
    print(f"Subset 180: {len(subset_180)} entries from {len({v['scene']['group_id'] for v in subset_180.values()})} groups")

    rng_split = random.Random(RANDOM_SEED)
    subset_cal, subset_val = split_subset_cal_val(subset_180, cal_ratio=0.25, rng=rng_split)
    dump_json(subset_cal, base_dir / "agument_with_img_unique_subset_180_cal.json")
    dump_json(subset_val, base_dir / "agument_with_img_unique_subset_180_val.json")
    print(f"Subset calibration: {len(subset_cal)} entries")
    print(f"Subset validation: {len(subset_val)} entries")


if __name__ == "__main__":
    main()
