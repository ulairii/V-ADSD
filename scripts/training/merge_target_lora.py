#!/usr/bin/env python3
"""Merge one complete DriveLM LoRA checkpoint into a standalone Qwen VLM."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-step", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    complete = args.output_dir / "complete.json"
    if complete.is_file() and (args.output_dir / "config.json").is_file():
        payload = json.loads(complete.read_text())
        if int(payload.get("source_step", -1)) != args.source_step:
            raise RuntimeError(f"existing export has different source step: {payload}")
        print(f"existing_complete_export={args.output_dir}", flush=True)
        return
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite incomplete output: {args.output_dir}")
    if not (args.adapter_checkpoint / "COMPLETE").is_file():
        raise RuntimeError(f"adapter checkpoint is not complete: {args.adapter_checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_dir.parent / f".{args.output_dir.name}.{os.getpid()}.tmp"
    temporary.mkdir(parents=False, exist_ok=False)

    processor = AutoProcessor.from_pretrained(
        args.base_model,
        local_files_only=True,
    )
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter_checkpoint, is_trainable=False)
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(temporary, safe_serialization=True, max_shard_size="5GB")
    processor.save_pretrained(temporary)
    (temporary / "complete.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "source_step": args.source_step,
                "base_model": str(args.base_model),
                "adapter_checkpoint": str(args.adapter_checkpoint),
                "torch_dtype": "bfloat16",
            },
            indent=2,
        )
        + "\n"
    )
    os.replace(temporary, args.output_dir)
    print(f"merged_target={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
