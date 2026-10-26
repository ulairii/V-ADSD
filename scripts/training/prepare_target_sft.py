#!/usr/bin/env python3
"""Prepare scene-disjoint DriveLM SFT manifests for a VLM target.

Two explicitly named protocols are supported:

* ``challenge`` reproduces the public DriveLM challenge extraction rules.
* ``all`` keeps every public QA, matching the paper's statement that all QAs
  are used, but cannot reconstruct graph contexts because the public v1.1
  context/edge fields are empty.

Large JSONL outputs are intended for scratch storage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from prepare_drivelm_official import CAMERA_ORDER, official_qas, resolve_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--nuscenes-root", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", choices=("challenge", "all"), required=True)
    parser.add_argument("--validation-scenes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def excluded_scenes(path: Path) -> set[str]:
    return {
        json.loads(line)["scene_id"]
        for line in path.read_text().splitlines()
        if line.strip()
    }


def iter_qas(
    frame: dict[str, Any], protocol: str, seed: int, frame_key: str
) -> Iterable[tuple[str, dict[str, Any]]]:
    if protocol == "challenge":
        selected = official_qas(frame, seed, frame_key)
    else:
        selected = frame["QA"]
    for task in ("perception", "prediction", "planning", "behavior"):
        for qa in selected.get(task, []):
            yield task, qa


def main() -> None:
    args = parse_args()
    data = json.loads(args.annotation.read_text())
    heldout = excluded_scenes(args.exclude_manifest)
    eligible = sorted(set(data) - heldout)
    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    if len(eligible) <= args.validation_scenes:
        raise RuntimeError("not enough eligible scenes for the requested validation split")
    validation = set(eligible[: args.validation_scenes])
    training = set(eligible[args.validation_scenes :])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": args.output_dir / "train.jsonl",
        "validation": args.output_dir / "validation.jsonl",
    }
    temporary_paths = {name: path.with_suffix(".jsonl.tmp") for name, path in paths.items()}
    handles = {name: path.open("w") for name, path in temporary_paths.items()}
    counts: Counter[str] = Counter()
    context_nonempty = 0
    written = 0
    try:
        for scene_id in sorted(data):
            if scene_id in heldout:
                continue
            split = "validation" if scene_id in validation else "train"
            scene = data[scene_id]
            for frame_id in sorted(scene["key_frames"]):
                frame = scene["key_frames"][frame_id]
                frame_key = f"{scene_id}_{frame_id}"
                image_paths = [
                    str(resolve_image(args.nuscenes_root, frame["image_paths"][camera]))
                    for camera in CAMERA_ORDER
                ]
                missing = [path for path in image_paths if not Path(path).is_file()]
                if missing:
                    raise FileNotFoundError(missing[0])
                for qa_index, (task, qa) in enumerate(
                    iter_qas(frame, args.protocol, args.seed, frame_key)
                ):
                    context = qa.get("C")
                    if context:
                        context_nonempty += 1
                    row = {
                        "id": f"{scene_id}_{frame_id}_{task}_{qa_index}",
                        "scene_id": scene_id,
                        "frame_id": frame_id,
                        "task": task,
                        "question": qa["Q"],
                        "answer": qa["A"],
                        "context": context,
                        "camera_order": list(CAMERA_ORDER),
                        "image_paths": image_paths,
                    }
                    handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts[f"{split}_qas"] += 1
                    counts[f"{split}_{task}"] += 1
                    written += 1
                    if args.limit is not None and written >= args.limit:
                        break
                if args.limit is not None and written >= args.limit:
                    break
            if args.limit is not None and written >= args.limit:
                break
    finally:
        for handle in handles.values():
            handle.close()

    for split, path in paths.items():
        os.replace(temporary_paths[split], path)
    overlap = training & validation | training & heldout | validation & heldout
    if overlap:
        raise RuntimeError(f"scene leakage detected: {sorted(overlap)}")
    summary = {
        "annotation": str(args.annotation),
        "protocol": args.protocol,
        "seed": args.seed,
        "excluded_evaluation_scenes": len(heldout),
        "train_scenes": len(training),
        "validation_scenes": len(validation),
        "scene_overlap": [],
        "public_nonempty_context_rows": context_nonempty,
        "counts": dict(sorted(counts.items())),
        "sha256": {
            split: hashlib.sha256(path.read_bytes()).hexdigest()
            for split, path in paths.items()
        },
    }
    atomic_write(args.output_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
