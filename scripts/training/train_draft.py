#!/usr/bin/env python3
"""Exactly resumable, single-GPU DriveLM domain adaptation for a ViSpec draft."""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from vispec.model.cnets_ours import Model
from vispec.model.configs import EConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--initial-spec-model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--mtp-steps", type=int, default=1)
    parser.add_argument("--num-q", type=int, default=2)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--save-every-steps", type=int, default=64)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def find_lm_head(base_model: Path) -> torch.Tensor:
    index_path = base_model / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        shard = index["weight_map"]["lm_head.weight"]
        with safe_open(base_model / shard, framework="pt", device="cpu") as handle:
            return handle.get_tensor("lm_head.weight").float()
    single = base_model / "model.safetensors"
    if single.is_file():
        with safe_open(single, framework="pt", device="cpu") as handle:
            return handle.get_tensor("lm_head.weight").float()
    raise FileNotFoundError(f"could not locate lm_head.weight under {base_model}")


class FeatureDataset(Dataset):
    def __init__(self, paths: list[Path], max_length: int):
        self.paths = paths
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        raw = torch.load(self.paths[index], map_location="cpu", weights_only=False)
        hidden = raw["hidden_state"][: self.max_length]
        embeds = raw["inputs_embeds"][: self.max_length]
        loss_mask = raw["loss_mask"][: self.max_length].bool()
        image_mask = raw["image_mask"][: self.max_length].bool()
        if hidden.shape[0] < 2:
            raise RuntimeError(f"feature too short: {self.paths[index]}")
        target = torch.cat((hidden[1:], torch.zeros_like(hidden[:1])), dim=0)
        shifted_embeds = torch.cat((embeds[1:], torch.zeros_like(embeds[:1])), dim=0)
        shifted_loss = torch.cat((loss_mask[1:], torch.zeros_like(loss_mask[:1])), dim=0)
        return {
            "hidden_states": hidden,
            "inputs_embeds": shifted_embeds,
            "target": target,
            "loss_mask": shifted_loss,
            "image_mask": image_mask,
            "attention_mask": torch.ones(hidden.shape[0], dtype=torch.long),
        }


def collate_one(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if len(features) != 1:
        raise RuntimeError("ViSpec DriveLM adaptation supports batch size one")
    return {key: value.unsqueeze(0) for key, value in features[0].items()}


def distillation_loss(
    head: torch.nn.Linear,
    target_hidden: torch.Tensor,
    predictions: torch.Tensor,
    loss_mask: torch.Tensor,
    topk: int = 10,
) -> tuple[torch.Tensor, int, int]:
    mask = loss_mask.bool()
    if int(mask.sum()) == 0:
        raise RuntimeError("sample has no answer tokens after max-length truncation")
    student_logits = head(predictions[mask])
    with torch.no_grad():
        teacher_logits = head(target_hidden[mask])
        teacher_p = F.softmax(teacher_logits, dim=-1)
    student_p = F.softmax(student_logits, dim=-1)
    probability_l1 = torch.abs(student_p - teacher_p).sum(dim=-1).mean()
    topk_indices = torch.topk(teacher_p, k=topk, dim=-1).indices
    ranked_logits = student_logits.gather(-1, topk_indices)
    reverse = torch.flip(ranked_logits, dims=[-1])
    denominator = torch.flip(torch.logcumsumexp(reverse, dim=-1), dims=[-1])
    ranking_loss = -(ranked_logits - denominator).sum(dim=-1).mean()
    correct = int((student_logits.argmax(-1) == teacher_logits.argmax(-1)).sum())
    total = int(mask.sum())
    return 10.0 * probability_l1 + 0.1 * ranking_loss, correct, total


def complete_checkpoints(output_dir: Path) -> list[tuple[int, Path]]:
    checkpoints = []
    for path in output_dir.glob("checkpoint-step-*"):
        if not path.is_dir() or not (path / "COMPLETE").is_file():
            continue
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        if (path / "training_state.pt").is_file() and (path / "trainer_state.json").is_file():
            checkpoints.append((step, path))
    return sorted(checkpoints)


def save_checkpoint(
    output_dir: Path,
    model: Model,
    optimizer: AdamW,
    scheduler: LambdaLR,
    epoch: int,
    next_sample: int,
    global_step: int,
) -> Path:
    final = output_dir / f"checkpoint-step-{global_step}"
    temporary = output_dir / f".{final.name}.{os.getpid()}.incomplete"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "next_sample": next_sample,
        "global_step": global_step,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }
    state_tmp = temporary / "training_state.pt.tmp"
    torch.save(state, state_tmp)
    os.replace(state_tmp, temporary / "training_state.pt")
    atomic_json(
        temporary / "trainer_state.json",
        {"epoch": epoch, "next_sample": next_sample, "global_step": global_step},
    )
    (temporary / "COMPLETE").write_text("complete\n")
    if final.exists():
        shutil.rmtree(final)
    os.replace(temporary, final)
    atomic_json(output_dir / "latest.json", {"checkpoint": str(final), "global_step": global_step})
    print(f"saved resumable checkpoint {final}", flush=True)
    return final


def export_model(output_dir: Path, model: Model, config_path: Path) -> Path:
    export_dir = output_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    tensors = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    temporary = export_dir / "model.safetensors.tmp"
    save_file(tensors, temporary)
    os.replace(temporary, export_dir / "model.safetensors")
    shutil.copyfile(config_path, export_dir / "config.json")
    return export_dir


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    paths = sorted(args.data_dir.rglob("*.ckpt"))
    if len(paths) < 2:
        raise RuntimeError(f"need at least two feature files, found {len(paths)}")
    random.Random(args.seed).shuffle(paths)
    validation_size = max(1, round(len(paths) * args.validation_fraction))
    train_paths, validation_paths = paths[:-validation_size], paths[-validation_size:]
    train_loader = DataLoader(
        FeatureDataset(train_paths, args.max_length),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_one,
    )
    validation_loader = DataLoader(
        FeatureDataset(validation_paths, args.max_length),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_one,
    )

    config = EConfig.from_pretrained(args.config)
    model = Model(config, load_emb=True, path=str(args.base_model), num_q=args.num_q)
    initial_weights = load_file(args.initial_spec_model / "model.safetensors", device="cpu")
    missing, unexpected = model.load_state_dict(initial_weights, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected initial keys: {unexpected}")
    print(f"initial missing keys={missing}", flush=True)
    model = model.to(device="cuda", dtype=torch.bfloat16)
    model.train()
    head_weight = find_lm_head(args.base_model)
    head = torch.nn.Linear(head_weight.shape[1], head_weight.shape[0], bias=False)
    head.weight.data.copy_(head_weight)
    head.requires_grad_(False)
    head = head.to(device="cuda", dtype=torch.bfloat16).eval()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=args.learning_rate, betas=(0.9, 0.95))
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = len(train_loader)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))

    scheduler = LambdaLR(optimizer, lr_lambda)
    start_epoch = 0
    start_sample = 0
    global_step = 0
    checkpoints = complete_checkpoints(args.output_dir)
    if checkpoints:
        _, checkpoint = checkpoints[-1]
        saved = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch = int(saved["epoch"])
        start_sample = int(saved["next_sample"])
        global_step = int(saved["global_step"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        print(
            f"exact resume checkpoint={checkpoint} epoch={start_epoch} "
            f"next_sample={start_sample} global_step={global_step}",
            flush=True,
        )
    else:
        print("starting from released ViSpec checkpoint", flush=True)

    stop_requested = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        print(f"received signal {signum}; checkpointing after current step", flush=True)
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    last_epoch = start_epoch
    last_next_sample = start_sample
    for epoch in range(start_epoch, args.epochs):
        model.train()
        for sample_index, cpu_batch in enumerate(train_loader):
            if epoch == start_epoch and sample_index < start_sample:
                continue
            batch = {key: value.cuda(non_blocking=True) for key, value in cpu_batch.items()}
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                batch["hidden_states"],
                inputs_embeds=batch["inputs_embeds"],
                attention_mask=batch["attention_mask"],
                image_mask=batch["image_mask"],
            )
            predictions = [prediction]
            recurrent = prediction
            for _ in range(args.mtp_steps):
                recurrent = torch.cat(
                    (batch["hidden_states"][:, :1], recurrent[:, :-1]), dim=1
                )
                recurrent = model(
                    recurrent,
                    inputs_embeds=batch["inputs_embeds"],
                    attention_mask=batch["attention_mask"],
                    image_mask=batch["image_mask"],
                )
                predictions.append(recurrent)
            stacked_predictions = torch.cat(predictions, dim=0)
            repeated_target = batch["target"].expand(
                args.mtp_steps + 1, *batch["target"].shape
            ).flatten(0, 1)
            repeated_mask = batch["loss_mask"].expand(
                args.mtp_steps + 1, *batch["loss_mask"].shape
            ).flatten(0, 1)
            loss, correct, token_count = distillation_loss(
                head, repeated_target, stacked_predictions, repeated_mask
            )
            loss.backward()
            torch.nn.utils.clip_grad_value_(trainable, 0.5)
            optimizer.step()
            scheduler.step()
            global_step += 1
            next_epoch = epoch
            next_sample = sample_index + 1
            if next_sample >= len(train_loader):
                next_epoch, next_sample = epoch + 1, 0
            last_epoch, last_next_sample = next_epoch, next_sample
            print(
                f"train step={global_step}/{total_steps} epoch={epoch} "
                f"sample={sample_index} loss={loss.item():.6f} "
                f"top1={correct / token_count:.4f} lr={scheduler.get_last_lr()[0]:.3e}",
                flush=True,
            )
            should_save = global_step % args.save_every_steps == 0 or stop_requested
            if should_save:
                save_checkpoint(
                    args.output_dir,
                    model,
                    optimizer,
                    scheduler,
                    next_epoch,
                    next_sample,
                    global_step,
                )
            if stop_requested:
                raise SystemExit(143)
            if args.max_steps is not None and global_step >= args.max_steps:
                break
        if args.max_steps is not None and global_step >= args.max_steps:
            break
        start_sample = 0

        model.eval()
        validation_loss = 0.0
        validation_correct = 0
        validation_tokens = 0
        with torch.inference_mode():
            for cpu_batch in validation_loader:
                batch = {key: value.cuda(non_blocking=True) for key, value in cpu_batch.items()}
                prediction = model(
                    batch["hidden_states"],
                    inputs_embeds=batch["inputs_embeds"],
                    attention_mask=batch["attention_mask"],
                    image_mask=batch["image_mask"],
                )
                loss, correct, token_count = distillation_loss(
                    head, batch["target"], prediction, batch["loss_mask"]
                )
                validation_loss += float(loss)
                validation_correct += correct
                validation_tokens += token_count
        metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "validation_loss": validation_loss / len(validation_loader),
            "validation_top1": validation_correct / validation_tokens,
            "validation_tokens": validation_tokens,
        }
        atomic_json(args.output_dir / f"metrics-epoch-{epoch}.json", metrics)
        print(json.dumps(metrics), flush=True)

    save_checkpoint(
        args.output_dir,
        model,
        optimizer,
        scheduler,
        last_epoch,
        last_next_sample,
        global_step,
    )
    export_dir = export_model(args.output_dir, model, args.config)
    atomic_json(
        args.output_dir / "complete.json",
        {
            "status": "complete",
            "global_step": global_step,
            "export_dir": str(export_dir),
            "train_samples": len(train_paths),
            "validation_samples": len(validation_paths),
        },
    )
    print(f"training complete export={export_dir}", flush=True)


if __name__ == "__main__":
    main()
