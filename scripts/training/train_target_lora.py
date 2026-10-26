#!/usr/bin/env python3
"""LoRA-SFT Qwen2.5-VL on a DriveLM JSONL manifest with exact resume."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import re
import signal
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


SYSTEM_PROMPT = (
    "You are an autonomous-driving visual question answering assistant. "
    "Answer using only the supplied surround-camera views. Return only the concise answer."
)
CAMERA_ORDER_TEXT = (
    "Six surround-camera views follow in this order: CAM_FRONT, CAM_FRONT_LEFT, "
    "CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--save-every-steps", type=int, default=250)
    parser.add_argument("--validation-samples", type=int, default=64)
    parser.add_argument("--max-visual-tokens-per-image", type=int, default=256)
    parser.add_argument("--max-answer-tokens", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--export-merged", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-step-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def complete_checkpoints(root: Path) -> list[Path]:
    required = {
        "COMPLETE",
        "adapter_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "training_state.pt",
    }
    output = []
    for path in root.glob("checkpoint-step-*"):
        if not path.is_dir() or not required.issubset({item.name for item in path.iterdir()}):
            print(f"skip_incomplete_checkpoint={path}", flush=True)
            continue
        if not ((path / "adapter_model.safetensors").is_file() or (path / "adapter_model.bin").is_file()):
            print(f"skip_incomplete_checkpoint={path}", flush=True)
            continue
        output.append(path)
    return sorted(output, key=checkpoint_step)


def capture_rng(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device),
    }


def restore_rng(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda_state = state["cuda"]
    if isinstance(cuda_state, list):
        cuda_state = cuda_state[device.index or 0]
    torch.cuda.set_rng_state(cuda_state, device)


def save_checkpoint(
    model: PeftModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    root: Path,
    state: dict[str, Any],
    rank: int,
    world_size: int,
    device: torch.device,
) -> Path:
    target = root / f"checkpoint-step-{state['global_step']}"
    local_rng = capture_rng(device)
    rng_by_rank: list[Any] = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(rng_by_rank, local_rng)
    else:
        rng_by_rank[0] = local_rng
    if rank == 0 and not (target.is_dir() and (target / "COMPLETE").is_file()):
        temporary = root / f".{target.name}.{os.getpid()}.tmp"
        temporary.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(temporary, safe_serialization=True)
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
        torch.save(
            {**state, "world_size": world_size, "rng_by_rank": rng_by_rank},
            temporary / "training_state.pt",
        )
        (temporary / "COMPLETE").write_text("complete\n")
        os.replace(temporary, target)
        print(f"saved_complete_checkpoint={target}", flush=True)
    if world_size > 1:
        dist.barrier()
    return target


def make_inputs(
    processor: Any,
    row: dict[str, Any],
    device: torch.device,
    max_answer_tokens: int,
) -> dict[str, torch.Tensor]:
    content: list[dict[str, str]] = [{"type": "text", "text": CAMERA_ORDER_TEXT}]
    images = []
    for camera, image_path in zip(row["camera_order"], row["image_paths"], strict=True):
        content.extend((
            {"type": "text", "text": f"\n{camera}:"},
            {"type": "image"},
        ))
        with Image.open(image_path) as source:
            images.append(source.convert("RGB"))
    question = row["question"]
    if row.get("context"):
        question = f"Context:\n{row['context']}\n\nQuestion:\n{question}"
    content.append({"type": "text", "text": f"\nQuestion: {question}"})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(images=images, text=prompt, return_tensors="pt")
    answer_ids = processor.tokenizer(
        row["answer"] + processor.tokenizer.eos_token,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[:, :max_answer_tokens]
    prompt_length = inputs["input_ids"].shape[1]
    inputs["input_ids"] = torch.cat((inputs["input_ids"], answer_ids), dim=1)
    inputs["attention_mask"] = torch.cat(
        (inputs["attention_mask"], torch.ones_like(answer_ids)), dim=1
    )
    labels = inputs["input_ids"].clone()
    labels[:, :prompt_length] = -100
    inputs["labels"] = labels
    return {key: value.to(device) for key, value in inputs.items()}


@torch.no_grad()
def validation_loss(
    model: PeftModel,
    processor: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    max_answer_tokens: int,
    rank: int,
    world_size: int,
) -> float:
    model.eval()
    losses = []
    for row in rows[rank::world_size]:
        batch = make_inputs(processor, row, device, max_answer_tokens)
        losses.append(float(model(**batch).loss.detach().cpu()))
    model.train()
    totals = torch.tensor([sum(losses), len(losses)], dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return float(totals[0].item() / max(totals[1].item(), 1))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    process_seed = args.seed + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed(process_seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    train_rows = load_rows(args.train_manifest)
    validation_rows = load_rows(args.validation_manifest)[: args.validation_samples]
    if not train_rows or not validation_rows:
        raise RuntimeError("training and validation manifests must be non-empty")

    processor = AutoProcessor.from_pretrained(
        args.base_model,
        min_pixels=256 * 28 * 28,
        max_pixels=args.max_visual_tokens_per_image * 28 * 28,
        local_files_only=True,
    )
    latest = complete_checkpoints(args.output_dir)
    resume = latest[-1] if latest else None
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if resume:
        peft_model = PeftModel.from_pretrained(base, resume, is_trainable=True)
    else:
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        )
        peft_model = get_peft_model(base, config)
    peft_model.enable_input_require_grads()
    peft_model.train()
    if rank == 0:
        peft_model.print_trainable_parameters()
    parameters = [parameter for parameter in peft_model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    state = {"epoch": 0, "next_position": 0, "global_step": 0}
    if resume:
        optimizer.load_state_dict(torch.load(resume / "optimizer.pt", map_location="cpu"))
        scheduler.load_state_dict(torch.load(resume / "scheduler.pt", map_location="cpu"))
        saved = torch.load(resume / "training_state.pt", map_location="cpu", weights_only=False)
        saved_world_size = int(saved.get("world_size", 1))
        if saved_world_size != world_size:
            raise RuntimeError(
                f"checkpoint world_size={saved_world_size} cannot exactly resume with world_size={world_size}"
            )
        state.update({key: saved[key] for key in state})
        rng_by_rank = saved.get("rng_by_rank")
        if rng_by_rank is None:
            rng_by_rank = [saved["rng"]]
        restore_rng(rng_by_rank[rank], device)
    if world_size > 1:
        train_model: Any = DistributedDataParallel(
            peft_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    else:
        train_model = peft_model
    if rank == 0:
        print(
            f"distributed_world_size={world_size} local_gradient_accumulation={args.gradient_accumulation}",
            flush=True,
        )
        print(f"resume_from={resume or 'none'} state={state}", flush=True)

    stop_requested = False
    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        if rank == 0:
            print(f"received_signal={signum}; checkpointing at next optimizer boundary", flush=True)
        stop_requested = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    optimizer.zero_grad(set_to_none=True)
    while state["epoch"] < args.epochs:
        indices = list(range(len(train_rows)))
        random.Random(args.seed + state["epoch"]).shuffle(indices)
        usable = len(indices) - (len(indices) % world_size)
        indices = indices[rank:usable:world_size]
        position = state["next_position"]
        while position < len(indices):
            accumulation_start = position
            micro_count = min(args.gradient_accumulation, len(indices) - position)
            for micro_index in range(micro_count):
                row = train_rows[indices[position]]
                batch = make_inputs(processor, row, device, args.max_answer_tokens)
                sync_context = (
                    train_model.no_sync()
                    if world_size > 1 and micro_index < micro_count - 1
                    else contextlib.nullcontext()
                )
                with sync_context:
                    loss = train_model(**batch).loss / micro_count
                    loss.backward()
                position += 1
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state["global_step"] += 1
            state["next_position"] = position
            loss_for_log = torch.tensor(float(loss.detach()) * micro_count, device=device)
            if world_size > 1:
                dist.all_reduce(loss_for_log, op=dist.ReduceOp.SUM)
                loss_for_log /= world_size
            if rank == 0 and state["global_step"] % 10 == 0:
                print(
                    f"epoch={state['epoch']} rows={position * world_size}/{usable} "
                    f"step={state['global_step']} loss={loss_for_log.item():.6f}",
                    flush=True,
                )
            should_save = state["global_step"] % args.save_every_steps == 0
            reached_cap = args.max_steps is not None and state["global_step"] >= args.max_steps
            if world_size > 1:
                stop_tensor = torch.tensor(int(stop_requested), device=device)
                dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
                stop_requested = bool(stop_tensor.item())
            if should_save or stop_requested or reached_cap:
                save_checkpoint(
                    peft_model, optimizer, scheduler, args.output_dir, dict(state),
                    rank, world_size, device,
                )
                if rank == 0:
                    atomic_json(args.output_dir / "progress.json", state)
            if stop_requested:
                if world_size > 1:
                    dist.destroy_process_group()
                raise SystemExit(143)
            if reached_cap:
                if rank == 0:
                    print(f"max_steps_reached={args.max_steps}", flush=True)
                if world_size > 1:
                    dist.destroy_process_group()
                return
            if position == accumulation_start:
                raise RuntimeError("training made no progress")
        state["epoch"] += 1
        state["next_position"] = 0
        val = validation_loss(
            peft_model, processor, validation_rows, device, args.max_answer_tokens,
            rank, world_size,
        )
        state["validation_loss"] = val
        if rank == 0:
            print(f"completed_epoch={state['epoch']} validation_loss={val:.6f}", flush=True)
        save_checkpoint(
            peft_model, optimizer, scheduler, args.output_dir, dict(state),
            rank, world_size, device,
        )
        if rank == 0:
            atomic_json(args.output_dir / "progress.json", state)

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        adapter_root = args.output_dir / "adapter"
        peft_model.save_pretrained(adapter_root, safe_serialization=True)
        processor.save_pretrained(adapter_root)
        if args.export_merged:
            merged_root = args.output_dir / "merged_target"
            merged = peft_model.merge_and_unload()
            merged.config.use_cache = True
            merged.save_pretrained(merged_root, safe_serialization=True, max_shard_size="5GB")
            processor.save_pretrained(merged_root)
            print(f"merged_target={merged_root}", flush=True)
        atomic_json(args.output_dir / "complete.json", {**state, "status": "complete"})
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
