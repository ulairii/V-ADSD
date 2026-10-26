#!/usr/bin/env python3
"""CPU-only integrity checks for the released V-ADSD artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-root",
        type=Path,
        help="optionally verify every manifest image against a nuScenes leaf or bundle root",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def close(actual: float, expected: float, tolerance: float = 1e-9) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"expected {expected}, got {actual}")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    protocol = json.loads((ROOT / "configs/paper_protocol.json").read_text())
    evidence = json.loads((ROOT / "results/paper_results.json").read_text())
    test = load_jsonl(ROOT / protocol["split"]["test_manifest"])
    calibration = load_jsonl(ROOT / protocol["split"]["calibration_manifest"])

    assert len(test) == protocol["split"]["test_questions"] == 51
    assert len(calibration) == protocol["split"]["calibration_questions"] == 10
    test_scenes = {row["scene_id"] for row in test}
    calibration_scenes = {row["scene_id"] for row in calibration}
    assert len(test_scenes) == protocol["split"]["test_scenes"] == 21
    assert len(calibration_scenes) == protocol["split"]["calibration_scenes"] == 10
    assert not test_scenes & calibration_scenes

    all_rows = test + calibration
    assert len({row["id"] for row in all_rows}) == len(all_rows)
    assert {row["task"] for row in all_rows} == {"planning"}
    for row in all_rows:
        assert len(row["image_paths"]) == 6
        assert all(Path(path).parts[0] == "samples" for path in row["image_paths"])
        assert len(row["reference_answer"].split()) >= 24

    if args.image_root is not None:
        missing = []
        for row in all_rows:
            for relative in row["image_paths"]:
                direct = args.image_root / relative
                nested = args.image_root / "data_full" / "nuscenes" / relative
                if not direct.is_file() and not nested.is_file():
                    missing.append(relative)
        if missing:
            raise AssertionError(f"missing {len(missing)} manifest images; first={missing[0]}")

    by_name = {row["condition"]: row for row in evidence["results"]}
    assert set(by_name) == {
        "clean", "random_sign_norm_matched", "fgsm_lambda1",
        "pgd_unconstrained", "pgd_lambda1", "pgd_lambda4",
    }
    clean = by_name["clean"]
    for name, row in by_name.items():
        if name == "clean":
            continue
        eta_delta = 100.0 * (row["eta"] / clean["eta"] - 1.0)
        latency_delta = 100.0 * (row["spec_total_sec"] / clean["spec_total_sec"] - 1.0)
        if name in {"pgd_unconstrained", "pgd_lambda1", "pgd_lambda4"}:
            close(eta_delta, row["eta_change_percent"], tolerance=0.06)
            close(latency_delta, row["spec_total_change_percent"], tolerance=0.06)
        else:
            close(eta_delta, row["eta_change_percent"], tolerance=1e-9)
            close(latency_delta, row["spec_total_change_percent"], tolerance=1e-9)

    primary = by_name["pgd_lambda1"]
    close(primary["eta_change_percent"], -42.0)
    close(primary["spec_total_change_percent"], 18.2)
    assert primary["target_text_exact"] == 50
    assert primary["target_action_exact"] == 51
    assert primary["spec_target_exact"] == 51

    print("V-ADSD artifact integrity: PASS")
    print(f"  test: {len(test)} questions / {len(test_scenes)} scenes")
    print(f"  calibration: {len(calibration)} questions / {len(calibration_scenes)} scenes")
    print(f"  test manifest sha256: {digest(ROOT / protocol['split']['test_manifest'])}")
    if args.image_root is not None:
        print(f"  image files: PASS ({len(all_rows) * 6} references)")
    print("  primary result: eta -42.0%, E2E latency +18.2%, action 51/51, SD match 51/51")
    print("  evidence level: aggregate result validation (GPU rerun required for independent reproduction)")


if __name__ == "__main__":
    main()
