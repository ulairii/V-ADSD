#!/usr/bin/env python3
"""Render the committed aggregate paper results as a Markdown table."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "paper_results.json"

LABELS = {
    "clean": "Clean",
    "random_sign_norm_matched": "Random sign",
    "fgsm_lambda1": "FGSM, λ=1",
    "pgd_unconstrained": "PGD, λ=0",
    "pgd_lambda1": "PGD, λ=1",
    "pgd_lambda4": "PGD, λ=4",
}


def count(value: int | None) -> str:
    return "—" if value is None else f"{value}/51"


def delta(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:+.1f}%".replace("-", "−")


def main() -> None:
    payload = json.loads(RESULTS.read_text())
    rows = payload["results"]
    header = (
        "| Condition | Tokens / target call | Change | E2E latency (s) | "
        "Change | Target F1 | Clean target match | SD / target match |"
    )
    print(header)
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {LABELS[row['condition']]} | {row['eta']:.3f} | "
            f"{delta(row.get('eta_change_percent'))} | "
            f"{row['spec_total_sec']:.3f} | "
            f"{delta(row.get('spec_total_change_percent'))} | "
            f"{row['target_f1']:.3f} | {count(row.get('target_text_exact'))} | "
            f"{count(row.get('spec_target_exact'))} |"
        )


if __name__ == "__main__":
    main()
