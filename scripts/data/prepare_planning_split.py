#!/usr/bin/env python3
"""Create a scene-disjoint planning split from a frozen DriveLM result file.

The frozen clean result is used only as a manifest source. Target/speculative
outputs are removed from the attack manifests; a compact audit retains the
clean text and deterministic action-term extraction for manual review.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


TERM_PATTERNS = {
    "stop": r"\b(?:stop|stationary|stopped|brake)\b",
    "forward": r"\b(?:keep going|continue ahead|go forward|going ahead|go straight)\b",
    "left": r"\b(?:turn|steer|steering|offset)\w*\s+to\s+the\s+left\b|\bleft\s+turn\b|\bchanging\s+to\s+the\s+left\s+lane\b",
    "right": r"\b(?:turn|steer|steering|offset)\w*\s+to\s+the\s+right\b|\bright\s+turn\b|\bchanging\s+to\s+the\s+right\s+lane\b",
    "back": r"\b(?:back up|reverse|backward)\b",
    "slow": r"\b(?:slow|decelerat|same speed)\w*\b",
    "accelerate": r"\baccelerat\w*\b",
    "collision": r"\bcollision\b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-scenes", type=int, default=6)
    return parser.parse_args()


def terms(text: str) -> list[str]:
    value = text.lower()
    return sorted(name for name, pattern in TERM_PATTERNS.items() if re.search(pattern, value))


def manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "id",
        "scene_id",
        "frame_id",
        "task",
        "question",
        "reference_answer",
        "tag",
        "camera_order",
        "image_paths",
        "image_path",
    )
    output = {key: row[key] for key in keep if key in row}
    if "reference_answer" not in output:
        output["reference_answer"] = row.get("target", {}).get("reference_answer", "")
    return output


def main() -> None:
    args = parse_args()
    rows = json.loads(args.clean_results.read_text())
    planning = [row for row in rows if row.get("task") == "planning"]
    if len(planning) != 42:
        raise RuntimeError(f"expected 42 planning rows, found {len(planning)}")
    scenes: list[str] = []
    for row in planning:
        scene = str(row["scene_id"])
        if scene not in scenes:
            scenes.append(scene)
    if len(scenes) < args.calibration_scenes + 1:
        raise RuntimeError("not enough scenes for a disjoint split")
    calibration_scene_set = set(scenes[: args.calibration_scenes])
    calibration = [row for row in planning if row["scene_id"] in calibration_scene_set]
    test = [row for row in planning if row["scene_id"] not in calibration_scene_set]
    reason_rows = [
        row for row in planning if "why take this action" in row["question"].lower()
    ]
    reason_calibration = [
        row for row in reason_rows if row["scene_id"] in calibration_scene_set
    ]
    reason_test = [
        row for row in reason_rows if row["scene_id"] not in calibration_scene_set
    ]
    if not calibration or not test:
        raise RuntimeError("empty calibration or test split")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in (
        ("planning_calibration.jsonl", calibration),
        ("planning_test.jsonl", test),
        ("planning_reason_calibration.jsonl", reason_calibration),
        ("planning_reason_test.jsonl", reason_test),
    ):
        path = args.output_dir / name
        payload = "".join(json.dumps(manifest_row(row), ensure_ascii=False) + "\n" for row in subset)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(payload)
        temporary.replace(path)

    audit_rows = []
    for row in planning:
        clean_text = row.get("target", {}).get("text", "")
        reference = row.get("target", {}).get("reference_answer", row.get("reference_answer", ""))
        audit_rows.append(
            {
                "id": row["id"],
                "scene_id": row["scene_id"],
                "task": row["task"],
                "split": "calibration" if row["scene_id"] in calibration_scene_set else "test",
                "generated_tokens": len(row.get("target", {}).get("token_ids", [])),
                "clean_text": clean_text,
                "reference_answer": reference,
                "clean_terms": terms(clean_text),
                "reference_terms": terms(reference),
            }
        )
    audit = {
        "source": str(args.clean_results),
        "num_planning": len(planning),
        "num_calibration": len(calibration),
        "num_test": len(test),
        "num_reason_planning": len(reason_rows),
        "num_reason_calibration": len(reason_calibration),
        "num_reason_test": len(reason_test),
        "calibration_scenes": scenes[: args.calibration_scenes],
        "test_scenes": scenes[args.calibration_scenes :],
        "task_counts": dict(Counter(row["task"] for row in planning)),
        "truncated_rows": [row["id"] for row in planning if len(row.get("target", {}).get("token_ids", [])) >= 64],
        "rows": audit_rows,
    }
    audit_path = args.output_dir / "planning_action_audit.json"
    temporary = audit_path.with_suffix(audit_path.suffix + ".tmp")
    temporary.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(audit_path)
    print(json.dumps({key: audit[key] for key in audit if key != "rows"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
