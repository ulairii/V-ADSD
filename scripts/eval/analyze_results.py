#!/usr/bin/env python3
"""Produce auditable latency/parity statistics for DriveLM ViSpec runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-results", type=Path, required=True)
    parser.add_argument("--paired-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def describe(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std_population": float(array.std()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
        "sum": float(array.sum()),
    }


def latency(rows: list[dict[str, Any]], decoder: str) -> dict[str, Any]:
    outputs = [row[decoder] for row in rows]
    return {
        "total_time_sec": describe([row["total_time_sec"] for row in outputs]),
        "post_prefill_time_sec": describe([row["decode_time_sec"] for row in outputs]),
        "generated_tokens": describe([len(row["token_ids"]) for row in outputs]),
    }


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    target_rows = json.loads(args.target_results.read_text())
    paired_rows = json.loads(args.paired_results.read_text())
    target_by_id = {row["id"]: row for row in target_rows}
    paired_by_id = {row["id"]: row for row in paired_rows}
    if set(target_by_id) != set(paired_by_id):
        raise ValueError("target and paired runs do not contain the same IDs")

    within_mismatches = [
        row["id"]
        for row in paired_rows
        if row["target"]["token_ids"] != row["spec"]["token_ids"]
    ]
    across_mismatches = [
        row_id
        for row_id in target_by_id
        if target_by_id[row_id]["target"]["token_ids"]
        != paired_by_id[row_id]["target"]["token_ids"]
    ]
    target_total = sum(row["target"]["total_time_sec"] for row in paired_rows)
    spec_total = sum(row["spec"]["total_time_sec"] for row in paired_rows)
    target_decode = sum(row["target"]["decode_time_sec"] for row in paired_rows)
    spec_decode = sum(row["spec"]["decode_time_sec"] for row in paired_rows)
    acceptance = [
        float(value)
        for row in paired_rows
        for value in row["spec"]["acceptance_length"]
    ]
    verifier_steps = sum(row["spec"]["verifier_steps"] for row in paired_rows)
    generated = sum(len(row["spec"]["token_ids"]) for row in paired_rows)
    by_task: dict[str, Any] = {}
    for task in sorted({row["task"] for row in paired_rows}):
        task_rows = [row for row in paired_rows if row["task"] == task]
        task_target_total = sum(row["target"]["total_time_sec"] for row in task_rows)
        task_spec_total = sum(row["spec"]["total_time_sec"] for row in task_rows)
        task_target_decode = sum(row["target"]["decode_time_sec"] for row in task_rows)
        task_spec_decode = sum(row["spec"]["decode_time_sec"] for row in task_rows)
        task_acceptance = [
            float(value)
            for row in task_rows
            for value in row["spec"]["acceptance_length"]
        ]
        task_verifier_steps = sum(row["spec"]["verifier_steps"] for row in task_rows)
        task_generated = sum(len(row["spec"]["token_ids"]) for row in task_rows)
        by_task[task] = {
            "num_samples": len(task_rows),
            "mean_generated_tokens": task_generated / len(task_rows),
            "end_to_end_speedup": task_target_total / task_spec_total,
            "post_prefill_loop_speedup": task_target_decode / task_spec_decode,
            "mean_raw_accept_length": sum(task_acceptance) / len(task_acceptance),
            "generated_tokens_per_verifier_step": task_generated / task_verifier_steps,
        }

    output = {
        "num_samples": len(paired_rows),
        "independent_target": latency(target_rows, "target"),
        "paired_target": latency(paired_rows, "target"),
        "paired_spec": latency(paired_rows, "spec"),
        "parity": {
            "target_vs_spec_rate": 1.0 - len(within_mismatches) / len(paired_rows),
            "target_vs_spec_mismatch_ids": within_mismatches,
            "independent_vs_paired_target_rate": 1.0
            - len(across_mismatches) / len(paired_rows),
            "independent_vs_paired_target_mismatch_ids": across_mismatches,
        },
        "aggregate_speedup": {
            "end_to_end": target_total / spec_total,
            "post_prefill_loop": target_decode / spec_decode,
        },
        "acceptance": {
            "raw_accept_length": describe(acceptance),
            "verifier_steps": verifier_steps,
            "generated_tokens": generated,
            "generated_tokens_per_verifier_step": generated / verifier_steps,
        },
        "by_task": by_task,
    }
    atomic_json(args.output, output)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
