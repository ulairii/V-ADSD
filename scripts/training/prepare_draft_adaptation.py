#!/usr/bin/env python3
"""Build a deterministic, scene-disjoint DriveLM draft-adaptation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

from prepare_drivelm_official import CAMERA_ORDER, official_qas, resolve_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--nuscenes-root", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=314159)
    return parser.parse_args()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    excluded_rows = [
        json.loads(line)
        for line in args.exclude_manifest.read_text().splitlines()
        if line.strip()
    ]
    excluded_scenes = {row["scene_id"] for row in excluded_rows}
    annotation = json.loads(args.annotation.read_text())
    candidates: list[dict[str, Any]] = []
    for scene_id in sorted(annotation):
        if scene_id in excluded_scenes:
            continue
        scene = annotation[scene_id]
        for frame_id in sorted(scene["key_frames"]):
            frame = scene["key_frames"][frame_id]
            key = f"{scene_id}_{frame_id}"
            qas = official_qas(frame, args.seed, key)
            image_paths = [
                str(resolve_image(args.nuscenes_root, frame["image_paths"][camera]))
                for camera in CAMERA_ORDER
            ]
            if not all(Path(path).is_file() for path in image_paths):
                continue
            flat_qas = [
                (task, qa)
                for task in ("perception", "prediction", "planning", "behavior")
                for qa in qas[task]
            ]
            for qa_index, (task, qa) in enumerate(flat_qas):
                candidates.append(
                    {
                        "id": f"{scene_id}_{frame_id}_{qa_index}",
                        "scene_id": scene_id,
                        "frame_id": frame_id,
                        "task": task,
                        "question": qa["Q"],
                        "reference_answer": qa["A"],
                        "tag": qa["tag"],
                        "camera_order": list(CAMERA_ORDER),
                        "image_paths": image_paths,
                    }
                )

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected = candidates[: args.count]
    if len(selected) != args.count:
        raise RuntimeError(f"requested {args.count} rows, found {len(selected)}")
    selected_scenes = {row["scene_id"] for row in selected}
    overlap = selected_scenes & excluded_scenes
    if overlap:
        raise RuntimeError(f"scene leakage: {sorted(overlap)}")

    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
    atomic_write(args.manifest, payload)
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    summary = {
        "annotation": str(args.annotation),
        "exclude_manifest": str(args.exclude_manifest),
        "seed": args.seed,
        "num_qas": len(selected),
        "num_scenes": len(selected_scenes),
        "excluded_scenes": len(excluded_scenes),
        "scene_overlap": [],
        "task_counts": {
            task: sum(row["task"] == task for row in selected)
            for task in ("perception", "prediction", "planning", "behavior")
        },
        "manifest_sha256": digest,
    }
    atomic_write(args.summary, json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
