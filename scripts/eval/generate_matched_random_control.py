#!/usr/bin/env python3
"""Create deterministic processed-tensor random-sign controls matched to attack norms."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from transformers import AutoProcessor

from run_drivelm_vispec import build_inputs, load_jsonl, safe_row_name
from optimize_visual_attack import atomic_torch_save, channel_vectors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--image-root",
        type=Path,
        help="Prefix for relative nuScenes image paths in the released manifest.",
    )
    parser.add_argument("--attack-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--indices")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-visual-tokens-per-image", type=int, default=256)
    return parser.parse_args()


def row_seed(seed: int, row_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{row_id}".encode()).digest()[:8], "little"
    ) & ((1 << 63) - 1)


def main() -> None:
    args = parse_args()
    rows = load_jsonl(
        args.manifest, args.limit, args.offset, args.indices, args.image_root
    )
    processor = AutoProcessor.from_pretrained(
        args.base_model,
        use_fast=True,
        min_pixels=256 * 28 * 28,
        max_pixels=args.max_visual_tokens_per_image * 28 * 28,
        local_files_only=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        row_id = row["id"]
        attack_path = args.attack_dir / f"{safe_row_name(row_id)}.pt"
        attack = torch.load(attack_path, map_location="cpu", weights_only=False)
        metadata = attack.get("metadata", {})
        budget = float(
            metadata.get(
                "projected_rgb_linf_255", metadata.get("rgb_linf_255", 0.0)
            )
        )
        if budget <= 0:
            raise ValueError(f"missing positive attack budget for {row_id}")
        inputs = build_inputs(processor, row)
        clean = inputs["pixel_values"].detach().float()
        mean, std = channel_vectors(processor, clean.shape[-1], clean.device)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(row_seed(args.seed, row_id))
        sign = torch.where(
            torch.rand(clean.shape, generator=generator) < 0.5,
            torch.tensor(-1.0),
            torch.tensor(1.0),
        ).to(clean.device)
        delta = sign * ((budget / 255.0) / std)
        lower = (0.0 - mean) / std
        upper = (1.0 - mean) / std
        random_pixels = torch.maximum(torch.minimum(clean + delta, upper), lower)
        out_metadata = {
            "method": "matched_random_sign",
            "seed": args.seed,
            "matched_budget_255": budget,
            "space": "Qwen normalized resized visual patches",
            "rgb_linf_255": float(((random_pixels - clean) * std).abs().max() * 255.0),
        }
        atomic_torch_save(
            args.output_dir / f"{safe_row_name(row_id)}.pt",
            {
                "row_id": row_id,
                "pixel_values": random_pixels.detach().cpu(),
                "image_grid_thw": inputs["image_grid_thw"].detach().cpu(),
                "metadata": out_metadata,
            },
        )
        print(f"generated matched control id={row_id} budget={budget:.4f}", flush=True)


if __name__ == "__main__":
    main()
