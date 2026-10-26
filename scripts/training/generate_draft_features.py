#!/usr/bin/env python3
"""Generate target rollout features for DriveLM-specific ViSpec adaptation."""

from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from run_drivelm_vispec import build_inputs, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-json", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-visual-tokens-per-image", type=int, default=256)
    parser.add_argument(
        "--torch-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def safe_name(index: int, row_id: str) -> str:
    clean = "".join(char if char.isalnum() or char in "-_" else "_" for char in row_id)
    return f"data_{index:05d}_{clean}.ckpt"


@torch.inference_mode()
def generate_feature(
    model: Any,
    processor: Any,
    row: dict[str, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    inputs = build_inputs(processor, row)
    prompt_length = int(inputs["input_ids"].shape[-1])
    outputs = model.generate(
        **inputs,
        output_hidden_states=True,
        return_dict_in_generate=True,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    input_embeds = torch.cat([step[0] for step in outputs.hidden_states], dim=1)
    target_hidden = torch.cat([step[-1] for step in outputs.hidden_states], dim=1)
    sequence = outputs.sequences
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    image_mask = (sequence == image_token_id)[..., :-1]
    loss_mask = torch.ones_like(sequence[:, :-1], dtype=torch.bool)
    loss_mask[:, : prompt_length - 1] = False
    if not (
        input_embeds.shape[1]
        == target_hidden.shape[1]
        == image_mask.shape[1]
        == loss_mask.shape[1]
    ):
        raise RuntimeError(
            "feature length mismatch: "
            f"embeds={input_embeds.shape} hidden={target_hidden.shape} "
            f"image_mask={image_mask.shape} loss_mask={loss_mask.shape}"
        )
    if int(loss_mask.sum()) < 1:
        raise RuntimeError("target generated no trainable token")
    return {
        "inputs_embeds": input_embeds.cpu()[0],
        "hidden_state": target_hidden.cpu()[0],
        "loss_mask": loss_mask.cpu()[0],
        "image_mask": image_mask.cpu()[0],
        "generated_ids": sequence[:, prompt_length:].cpu()[0],
        "row_id": row["id"],
        "scene_id": row["scene_id"],
        "task": row["task"],
        "prompt_length": prompt_length,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.manifest, args.limit)
    processor = AutoProcessor.from_pretrained(
        args.base_model,
        use_fast=True,
        min_pixels=256 * 28 * 28,
        max_pixels=args.max_visual_tokens_per_image * 28 * 28,
        local_files_only=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        torch_dtype=getattr(torch, args.torch_dtype),
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    stop_requested = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        print(f"received signal {signum}; stopping after current sample", flush=True)
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    completed = 0
    total_loss_tokens = 0
    for index, row in enumerate(rows):
        output = args.output_dir / safe_name(index, row["id"])
        if output.is_file():
            cached = torch.load(output, map_location="cpu", weights_only=False)
            completed += 1
            total_loss_tokens += int(cached["loss_mask"].sum())
            continue
        feature = generate_feature(model, processor, row, args.max_new_tokens)
        temporary = output.with_suffix(output.suffix + ".tmp")
        torch.save(feature, temporary)
        os.replace(temporary, output)
        completed += 1
        total_loss_tokens += int(feature["loss_mask"].sum())
        atomic_json(
            args.progress_json,
            {
                "status": "running",
                "completed": completed,
                "requested": len(rows),
                "loss_tokens": total_loss_tokens,
                "last_id": row["id"],
            },
        )
        print(
            f"feature {completed}/{len(rows)} id={row['id']} "
            f"prompt={feature['prompt_length']} loss_tokens={int(feature['loss_mask'].sum())}",
            flush=True,
        )
        if stop_requested:
            break
    status = "complete" if completed == len(rows) else "preempted"
    atomic_json(
        args.progress_json,
        {
            "status": status,
            "completed": completed,
            "requested": len(rows),
            "loss_tokens": total_loss_tokens,
        },
    )
    if status != "complete":
        raise SystemExit(143)


if __name__ == "__main__":
    main()
