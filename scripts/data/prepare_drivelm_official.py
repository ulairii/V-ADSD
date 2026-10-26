#!/usr/bin/env python3
"""Build a deterministic 100-QA subset using DriveLM challenge semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any


MOVING_CHOICES = [
    "Going ahead.",
    "Turn right.",
    "Turn left.",
    "Stopped.",
    "Back up.",
    "Reverse parking.",
    "Drive backward.",
]
BEHAVIOR_CHOICES = [
    "The ego vehicle is slightly steering to the left. The ego vehicle is driving very fast.",
    "The ego vehicle is steering to the left. The ego vehicle is driving with normal speed.",
    "The ego vehicle is steering to the left. The ego vehicle is driving fast.",
    "The ego vehicle is slightly steering to the right. The ego vehicle is driving fast.",
    "The ego vehicle is going straight. The ego vehicle is driving slowly.",
    "The ego vehicle is going straight. The ego vehicle is driving with normal speed.",
    "The ego vehicle is slightly steering to the left. The ego vehicle is driving with normal speed.",
    "The ego vehicle is slightly steering to the left. The ego vehicle is driving slowly.",
    "The ego vehicle is slightly steering to the right. The ego vehicle is driving slowly.",
    "The ego vehicle is slightly steering to the right. The ego vehicle is driving very fast.",
    "The ego vehicle is steering to the right. The ego vehicle is driving fast.",
    "The ego vehicle is steering to the right. The ego vehicle is driving very fast.",
    "The ego vehicle is slightly steering to the left. The ego vehicle is driving fast.",
    "The ego vehicle is steering to the left. The ego vehicle is driving very fast.",
    "The ego vehicle is going straight. The ego vehicle is not moving.",
    "The ego vehicle is slightly steering to the right. The ego vehicle is driving with normal speed.",
    "The ego vehicle is steering to the right. The ego vehicle is driving slowly.",
    "The ego vehicle is steering to the right. The ego vehicle is driving with normal speed.",
    "The ego vehicle is going straight. The ego vehicle is driving very fast.",
    "The ego vehicle is going straight. The ego vehicle is driving fast.",
    "The ego vehicle is steering to the left. The ego vehicle is driving slowly.",
]
CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--nuscenes-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--test-eval", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def multiple_choice(question: str, answer: str, pool: list[str], rng: random.Random) -> dict[str, Any]:
    if answer not in pool:
        raise ValueError(f"answer is absent from official choice pool: {answer!r}")
    alternatives = [choice for choice in pool if choice != answer]
    choices = rng.sample(alternatives, 3) + [answer]
    rng.shuffle(choices)
    letters = "ABCD"
    suffix = " ".join(f"{letters[i]}. {choice}" for i, choice in enumerate(choices))
    return {
        "Q": f"{question} Please select the correct answer from the following options: {suffix}",
        "A": letters[choices.index(answer)],
        "tag": [0],
    }


def official_qas(frame: dict[str, Any], seed: int, frame_key: str) -> dict[str, list[dict[str, Any]]]:
    source = frame["QA"]
    objects = frame["key_object_infos"]
    classes = [item["Visual_description"].split(".")[0] for item in objects.values()]
    locations = list(objects)
    output = {task: [] for task in ("perception", "prediction", "planning", "behavior")}

    for qa in source["perception"]:
        if all(name.lower() in qa["A"].lower() for name in classes):
            selected = dict(qa)
            selected["tag"] = [2]
            output["perception"].append(selected)
            break
    for qa_index, qa in enumerate(source["perception"]):
        if "what is the moving status of object" in qa["Q"].lower():
            rng = random.Random(f"{seed}:{frame_key}:perception:{qa_index}")
            output["perception"].append(
                multiple_choice(qa["Q"], qa["A"], MOVING_CHOICES, rng)
            )
            break

    for qa in source["prediction"]:
        if all(location.lower() in qa["A"].lower() for location in locations):
            selected = dict(qa)
            selected["tag"] = [3]
            output["prediction"].append(selected)
            break
    for qa in source["prediction"]:
        if "yes" in qa["A"].lower() or "no" in qa["A"].lower():
            selected = dict(qa)
            selected["tag"] = [0]
            output["prediction"].append(selected)
            break

    found = {"actions": False, "collision": False, "safe": False}
    for qa in source["planning"]:
        question = qa["Q"].lower()
        kind = None
        if "what actions could the ego vehicle take" in question and not found["actions"]:
            kind = "actions"
        elif "lead to a collision" in question and not found["collision"]:
            kind = "collision"
        elif "safe actions" in question and not found["safe"]:
            kind = "safe"
        if kind:
            selected = dict(qa)
            selected["tag"] = [1]
            output["planning"].append(selected)
            found[kind] = True
        if all(found.values()):
            break

    for qa_index, qa in enumerate(source.get("behavior", [])):
        rng = random.Random(f"{seed}:{frame_key}:behavior:{qa_index}")
        output["behavior"].append(
            multiple_choice(qa["Q"], qa["A"], BEHAVIOR_CHOICES, rng)
        )
    return output


def resolve_image(root: Path, relative: str) -> Path:
    relative = relative.replace("../nuscenes/", "").replace("data/nuscenes/", "")
    direct = root / relative
    nested = root / "data_full" / "nuscenes" / relative
    if direct.is_file() or not nested.is_file():
        return direct
    return nested


def select_exact(frames: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    reachable: dict[int, list[int]] = {0: []}
    for index, frame in enumerate(frames):
        size = frame["count"]
        for subtotal, indices in list(reachable.items())[::-1]:
            new_total = subtotal + size
            if new_total <= count and new_total not in reachable:
                reachable[new_total] = indices + [index]
        if count in reachable:
            return [frames[i] for i in reachable[count]]
    raise RuntimeError(f"could not select complete frames totaling {count} QA")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    data = json.loads(args.annotation.read_text())
    candidates: list[dict[str, Any]] = []
    for scene_id, scene in data.items():
        for frame_id, frame in scene["key_frames"].items():
            key = f"{scene_id}_{frame_id}"
            qas = official_qas(frame, args.seed, key)
            count = sum(len(items) for items in qas.values())
            if count:
                candidates.append(
                    {
                        "scene_id": scene_id,
                        "frame_id": frame_id,
                        "frame": frame,
                        "qas": qas,
                        "count": count,
                    }
                )
    random.Random(args.seed).shuffle(candidates)
    selected = select_exact(candidates, args.count)

    test_eval: dict[str, Any] = {}
    manifest: list[dict[str, Any]] = []
    for item in selected:
        scene_id, frame_id = item["scene_id"], item["frame_id"]
        frame = item["frame"]
        image_paths = [
            str(resolve_image(args.nuscenes_root, frame["image_paths"][camera]))
            for camera in CAMERA_ORDER
        ]
        missing = [path for path in image_paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(missing[0])
        test_eval.setdefault(scene_id, {"key_frames": {}})["key_frames"][frame_id] = {
            "QA": item["qas"],
            "image_paths": frame["image_paths"],
        }
        flat_qas = [
            (task, qa)
            for task in ("perception", "prediction", "planning", "behavior")
            for qa in item["qas"][task]
        ]
        for index, (task, qa) in enumerate(flat_qas):
            manifest.append(
                {
                    "id": f"{scene_id}_{frame_id}_{index}",
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

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest))
    os.replace(temporary, args.manifest)
    atomic_json(args.test_eval, test_eval)
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    atomic_json(
        args.summary,
        {
            "annotation": str(args.annotation),
            "seed": args.seed,
            "num_frames": len(selected),
            "num_qas": len(manifest),
            "task_counts": {
                task: sum(row["task"] == task for row in manifest)
                for task in ("perception", "prediction", "planning", "behavior")
            },
            "tag_counts": {
                str(tag): sum(tag in row["tag"] for row in manifest)
                for tag in range(4)
            },
            "manifest_sha256": manifest_hash,
        },
    )
    print(f"wrote {len(manifest)} QA from {len(selected)} complete frames sha256={manifest_hash}")


if __name__ == "__main__":
    main()
