#!/usr/bin/env python3
"""Optimize bounded visual inputs against DriveLM ViSpec acceptance surrogates.

The perturbation is applied to Qwen's normalized, resized visual patch tensor.
Projection uses the processor's channel mean/std, so the saved tensor has a
well-defined RGB-equivalent L-infinity budget after resizing. Model parameters
remain frozen; gradients are taken only with respect to the visual input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoProcessor

from run_drivelm_vispec import (
    atomic_json,
    build_inputs,
    load_jsonl,
    safe_row_name,
)
from vispec.model.spec_model_ours import SpecModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--image-root",
        type=Path,
        help="Prefix for relative nuScenes image paths in the released manifest.",
    )
    parser.add_argument("--clean-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-json", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--spec-model", required=True)
    parser.add_argument("--method", choices=("fgsm", "bim", "pgd"), default="pgd")
    parser.add_argument(
        "--objective",
        choices=("draft_ce", "topk_exclusion"),
        default="draft_ce",
        help=(
            "draft_ce preserves the original surrogate; topk_exclusion directly "
            "pushes target-path tokens below the draft tree's top-k boundary"
        ),
    )
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--step-size", type=float, default=2.0)
    parser.add_argument("--preservation-weight", type=float, default=1.0)
    parser.add_argument(
        "--preservation-objective",
        choices=("mean_ce", "max_ce", "hidden_mse"),
        default="mean_ce",
        help="aggregate clean-target teacher-forced CE over protected tokens",
    )
    parser.add_argument("--attack-tokens", type=int, default=4)
    parser.add_argument(
        "--attack-start",
        type=int,
        default=-1,
        help=(
            "generated-token index at which the attack starts (index 0 is the "
            "target root); -1 starts after the first decoded sentence"
        ),
    )
    parser.add_argument(
        "--preservation-tokens",
        type=int,
        help="number of clean target rollout tokens to constrain; 0 means all",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--indices",
        help="comma-separated zero-based manifest indices; overrides offset/limit",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-visual-tokens-per-image", type=int, default=256)
    parser.add_argument(
        "--torch-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    return parser.parse_args()


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_clean_tokens(path: Path) -> dict[str, list[int]]:
    rows = json.loads(path.read_text())
    clean: dict[str, list[int]] = {}
    for row in rows:
        target = row.get("target")
        if target and target.get("token_ids"):
            clean[row["id"]] = [int(token) for token in target["token_ids"]]
    if not clean:
        raise RuntimeError(f"no target token sequences found in {path}")
    return clean


def first_sentence_start(tokenizer: Any, token_ids: list[int]) -> int:
    """Return the first generated-token index after the action sentence."""
    for index in range(1, max(1, len(token_ids) - 1)):
        prefix = tokenizer.decode(token_ids[:index], skip_special_tokens=True)
        if "." in prefix:
            return index
    return 1


def channel_vectors(
    processor: Any, feature_dim: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    image_processor = processor.image_processor
    patch_size = int(image_processor.patch_size)
    temporal_patch_size = int(image_processor.temporal_patch_size)
    expected = temporal_patch_size * 3 * patch_size * patch_size
    if feature_dim != expected:
        raise RuntimeError(
            f"unexpected Qwen visual feature dimension {feature_dim}; expected {expected}"
        )
    mean = torch.tensor(image_processor.image_mean, dtype=torch.float32, device=device)
    std = torch.tensor(image_processor.image_std, dtype=torch.float32, device=device)
    # Qwen flattens each patch in (channel, temporal, height, width) order.
    values_per_channel = temporal_patch_size * patch_size * patch_size
    mean_vector = mean.repeat_interleave(values_per_channel)
    std_vector = std.repeat_interleave(values_per_channel)
    return mean_vector.unsqueeze(0), std_vector.unsqueeze(0)


def build_full_sequence_inputs(
    model: SpecModel,
    prompt_inputs: dict[str, torch.Tensor],
    generated_ids: list[int],
    pixel_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return target hidden states, multimodal inputs_embeds, and full IDs."""
    base = model.base_model
    prompt_ids = prompt_inputs["input_ids"]
    generated = torch.tensor(
        generated_ids, dtype=prompt_ids.dtype, device=prompt_ids.device
    ).unsqueeze(0)
    input_ids = torch.cat((prompt_ids, generated), dim=1)
    prompt_attention = prompt_inputs.get("attention_mask")
    if prompt_attention is None:
        prompt_attention = torch.ones_like(prompt_ids)
    attention_mask = torch.cat(
        (prompt_attention, torch.ones_like(generated, device=prompt_attention.device)),
        dim=1,
    )

    inputs_embeds = base.model.embed_tokens(input_ids)
    image_grid_thw = prompt_inputs["image_grid_thw"]
    image_embeds = base.visual(
        pixel_values.to(dtype=base.visual.dtype), grid_thw=image_grid_thw
    )
    image_mask = input_ids == base.config.image_token_id
    expanded_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
    if int(expanded_mask.sum()) != image_embeds.numel():
        raise RuntimeError(
            "image embedding/token mismatch: "
            f"mask={int(expanded_mask.sum())} embeds={image_embeds.numel()}"
        )
    inputs_embeds = inputs_embeds.masked_scatter(
        expanded_mask, image_embeds.to(inputs_embeds)
    )
    position_ids, _ = base.get_rope_index(
        input_ids,
        image_grid_thw,
        None,
        None,
        attention_mask,
    )
    outputs = base.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=None,
        inputs_embeds=inputs_embeds,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        cache_position=None,
    )
    return outputs.last_hidden_state, inputs_embeds, input_ids


def attack_objective(
    model: SpecModel,
    prompt_inputs: dict[str, torch.Tensor],
    generated_ids: list[int],
    pixel_values: torch.Tensor,
    attack_tokens: int,
    preservation_weight: float,
    objective_name: str,
    preservation_tokens: int | None,
    preservation_objective: str,
    clean_hidden: torch.Tensor | None,
    attack_start: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    hidden, inputs_embeds, input_ids = build_full_sequence_inputs(
        model, prompt_inputs, generated_ids, pixel_values
    )
    prompt_length = int(prompt_inputs["input_ids"].shape[1])
    history_hidden = hidden[:, :-1]
    shifted_embeds = inputs_embeds[:, 1:]
    image_mask = (input_ids == model.base_model.config.image_token_id)[:, :-1]
    attention_mask = torch.ones(
        history_hidden.shape[:2], dtype=torch.long, device=history_hidden.device
    )
    model.spec_layer.reset_kv()
    draft_hidden = model.spec_layer(
        history_hidden,
        inputs_embeds=shifted_embeds,
        attention_mask=attention_mask,
        image_mask=image_mask,
        use_cache=False,
    )

    # ViSpec first obtains generated_ids[0] from the target model as the root of
    # every candidate tree.  The draft's first attackable prediction is
    # generated_ids[1], conditioned on that root.  Starting one position earlier
    # attacks the target-only root rather than speculative acceptance.
    if attack_start < 1 or attack_start >= len(generated_ids):
        raise RuntimeError(
            f"invalid attack start {attack_start} for {len(generated_ids)} generated tokens"
        )
    count = min(attack_tokens, len(generated_ids) - attack_start)
    if count < 1:
        raise RuntimeError("clean target rollout has no draft-attackable token")
    start = prompt_length - 1 + attack_start - 1
    stop = start + count
    if start < 0 or stop > draft_hidden.shape[1]:
        raise RuntimeError(
            f"invalid attack positions [{start}:{stop}] for length {draft_hidden.shape[1]}"
        )
    head = model.base_model.lm_head
    draft_logits = head(draft_hidden[:, start:stop]).float()
    requested_preservation = attack_start if preservation_tokens is None else preservation_tokens
    preservation_count = (
        len(generated_ids)
        if requested_preservation == 0
        else min(requested_preservation, len(generated_ids))
    )
    target_logits = head(
        hidden[:, prompt_length - 1 : prompt_length - 1 + preservation_count]
    ).float()
    labels = torch.tensor(
        generated_ids[attack_start : attack_start + count],
        dtype=torch.long,
        device=draft_logits.device,
    ).unsqueeze(0)
    target_labels = torch.tensor(
        generated_ids[:preservation_count],
        dtype=torch.long,
        device=target_logits.device,
    ).unsqueeze(0)
    draft_ce = F.cross_entropy(
        draft_logits.reshape(-1, draft_logits.shape[-1]), labels.reshape(-1)
    )
    target_ce_per_token = F.cross_entropy(
        target_logits.reshape(-1, target_logits.shape[-1]),
        target_labels.reshape(-1),
        reduction="none",
    )
    target_ce_mean = target_ce_per_token.mean()
    target_ce_max = target_ce_per_token.max()
    hidden_mse = (
        F.mse_loss(hidden.float(), clean_hidden.float())
        if clean_hidden is not None
        else torch.zeros((), device=hidden.device)
    )
    if preservation_objective == "mean_ce":
        target_preservation_loss = target_ce_mean
    elif preservation_objective == "max_ce":
        target_preservation_loss = target_ce_max
    elif preservation_objective == "hidden_mse":
        if clean_hidden is None:
            raise ValueError("hidden_mse requires clean target hidden states")
        target_preservation_loss = hidden_mse
    else:
        raise ValueError(
            f"unsupported preservation objective: {preservation_objective}"
        )
    label_logits = draft_logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    competing_logits = draft_logits.scatter(
        -1, labels.unsqueeze(-1), torch.finfo(draft_logits.dtype).min
    )
    boundary_logits = torch.topk(
        competing_logits, k=int(model.spec_layer.top_k), dim=-1
    ).values[..., -1]
    exclusion_margin = (boundary_logits - label_logits).mean()
    if objective_name == "draft_ce":
        draft_attack = draft_ce
    elif objective_name == "topk_exclusion":
        draft_attack = exclusion_margin
    else:
        raise ValueError(f"unsupported objective: {objective_name}")
    objective = draft_attack - preservation_weight * target_preservation_loss
    metrics = {
        "objective": float(objective.detach()),
        "draft_ce": float(draft_ce.detach()),
        "draft_topk_exclusion_margin": float(exclusion_margin.detach()),
        "target_ce": float(target_preservation_loss.detach()),
        "target_ce_mean": float(target_ce_mean.detach()),
        "target_ce_max": float(target_ce_max.detach()),
        "target_hidden_mse": float(hidden_mse.detach()),
        "target_token_agreement": float(
            (target_logits.argmax(dim=-1) == target_labels).float().mean().detach()
        ),
        "draft_token_agreement": float(
            (draft_logits.argmax(dim=-1) == labels).float().mean().detach()
        ),
    }
    return objective, metrics


def seed_for_row(seed: int, row_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{row_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if not 0 < args.epsilon <= 32:
        raise ValueError("epsilon must be in (0, 32]")
    if args.method == "fgsm":
        args.steps = 1
        args.step_size = args.epsilon
    if args.steps < 1 or args.step_size <= 0:
        raise ValueError("steps and step-size must be positive")
    if args.preservation_tokens is not None and args.preservation_tokens < 0:
        raise ValueError("preservation-tokens must be non-negative")

    rows = load_jsonl(
        args.manifest, args.limit, args.offset, args.indices, args.image_root
    )
    clean_tokens = load_clean_tokens(args.clean_results)
    missing = [row["id"] for row in rows if row["id"] not in clean_tokens]
    if missing:
        raise RuntimeError(f"clean target results missing {len(missing)} requested rows")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(
        args.base_model,
        use_fast=True,
        min_pixels=256 * 28 * 28,
        max_pixels=args.max_visual_tokens_per_image * 28 * 28,
        local_files_only=True,
    )
    model = SpecModel.from_pretrained(
        base_model_path=args.base_model,
        spec_model_path=args.spec_model,
        total_token=30,
        depth=3,
        top_k=8,
        num_q=2,
        torch_dtype=getattr(torch, args.torch_dtype),
        low_cpu_mem_usage=True,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)
    model.base_model.config.use_cache = False
    if not args.disable_gradient_checkpointing and hasattr(
        model.base_model, "gradient_checkpointing_enable"
    ):
        model.base_model.gradient_checkpointing_enable()
        # The vendored Qwen implementation guards checkpoint calls with
        # ``self.training``. Keep every parameter frozen, but activate that
        # execution branch for the two large differentiable towers. Qwen2.5-VL
        # uses zero dropout in these modules, so this does not introduce attack
        # randomness.
        attention_dropout = float(
            getattr(model.base_model.config, "attention_dropout", 0.0)
        )
        if attention_dropout != 0.0:
            raise RuntimeError(
                "gradient-checkpoint attack requires zero attention dropout; "
                f"found {attention_dropout}"
            )
        model.base_model.visual.train()
        model.base_model.model.train()

    stop_requested = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        print(f"received signal {signum}; saving current attack step", flush=True)
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    completed = 0
    for row_index, row in enumerate(rows):
        output = args.output_dir / f"{safe_row_name(row['id'])}.pt"
        working = args.output_dir / f"{safe_row_name(row['id'])}.working.pt"
        if output.is_file():
            completed += 1
            continue
        prompt_inputs = build_inputs(processor, row)
        clean_pixels = prompt_inputs["pixel_values"].detach().float()
        attack_start = (
            first_sentence_start(model.tokenizer, clean_tokens[row["id"]])
            if args.attack_start < 0
            else args.attack_start
        )
        clean_hidden = None
        if args.preservation_objective == "hidden_mse":
            with torch.no_grad():
                clean_hidden, _, _ = build_full_sequence_inputs(
                    model,
                    prompt_inputs,
                    clean_tokens[row["id"]],
                    clean_pixels,
                )
                clean_hidden = clean_hidden.detach()
        mean_vector, std_vector = channel_vectors(
            processor, clean_pixels.shape[-1], clean_pixels.device
        )
        epsilon_vector = (args.epsilon / 255.0) / std_vector
        step_vector = (args.step_size / 255.0) / std_vector
        normalized_min = (0.0 - mean_vector) / std_vector
        normalized_max = (1.0 - mean_vector) / std_vector
        lower = torch.maximum(clean_pixels - epsilon_vector, normalized_min)
        upper = torch.minimum(clean_pixels + epsilon_vector, normalized_max)

        trace: list[dict[str, float]] = []
        start_step = 0
        if working.is_file():
            state = torch.load(working, map_location="cpu", weights_only=False)
            if (
                state.get("row_id") != row["id"]
                or state.get("method") != args.method
                or state.get("objective", "draft_ce") != args.objective
                or state.get("preservation_objective", "mean_ce")
                != args.preservation_objective
            ):
                raise RuntimeError(f"incompatible working state: {working}")
            adversarial = state["pixel_values"].to(clean_pixels.device).float()
            trace = list(state.get("trace", []))
            start_step = int(state.get("next_step", 0))
        elif args.method == "pgd":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed_for_row(args.seed, row["id"]))
            random_unit = torch.rand(
                clean_pixels.shape, generator=generator, dtype=torch.float32
            ).to(clean_pixels.device)
            adversarial = lower + random_unit * (upper - lower)
        else:
            adversarial = clean_pixels.clone()

        for step in range(start_step, args.steps):
            adversarial = adversarial.detach().requires_grad_(True)
            objective, metrics = attack_objective(
                model,
                prompt_inputs,
                clean_tokens[row["id"]],
                adversarial,
                args.attack_tokens,
                args.preservation_weight,
                args.objective,
                args.preservation_tokens,
                args.preservation_objective,
                clean_hidden,
                attack_start,
            )
            # The vendored Qwen towers use reentrant activation checkpointing,
            # which supports backward() but not autograd.grad(inputs=...). All
            # model parameters are frozen above, so only the visual leaf stores
            # a gradient here.
            objective.backward()
            gradient = adversarial.grad
            if gradient is None:
                raise RuntimeError(f"missing visual gradient for {row['id']} step {step}")
            if not torch.isfinite(gradient).all():
                raise RuntimeError(f"non-finite visual gradient for {row['id']} step {step}")
            with torch.no_grad():
                adversarial = adversarial + step_vector * gradient.sign()
                adversarial = torch.maximum(torch.minimum(adversarial, upper), lower)
            raw_delta = (adversarial - clean_pixels) * std_vector
            metrics.update(
                {
                    "step": float(step + 1),
                    "gradient_l1_mean": float(gradient.abs().mean()),
                    "rgb_linf_255": float(raw_delta.abs().max() * 255.0),
                }
            )
            trace.append(metrics)
            atomic_torch_save(
                working,
                {
                    "row_id": row["id"],
                    "method": args.method,
                    "objective": args.objective,
                    "preservation_objective": args.preservation_objective,
                    "next_step": step + 1,
                    "pixel_values": adversarial.detach().cpu(),
                    "trace": trace,
                    "attack_start": attack_start,
                },
            )
            atomic_json(
                args.progress_json,
                {
                    "status": "running",
                    "completed": completed,
                    "requested": len(rows),
                    "current_row": row_index,
                    "current_id": row["id"],
                    "current_step": step + 1,
                    "steps": args.steps,
                    "updated_unix": time.time(),
                },
            )
            print(
                f"attack row={row_index + 1}/{len(rows)} step={step + 1}/{args.steps} "
                f"id={row['id']} objective={metrics['objective']:.6f} "
                f"draft_ce={metrics['draft_ce']:.6f} target_ce={metrics['target_ce']:.6f} "
                f"linf={metrics['rgb_linf_255']:.3f}",
                flush=True,
            )
            if stop_requested:
                raise SystemExit(143)

        raw_delta = (adversarial - clean_pixels) * std_vector
        mse = float(raw_delta.square().mean())
        metadata = {
            "method": args.method,
            "objective": args.objective,
            "epsilon_255": args.epsilon,
            "step_size_255": args.step_size,
            "steps": args.steps,
            "preservation_weight": args.preservation_weight,
            "preservation_objective": args.preservation_objective,
            "preservation_tokens": (
                args.attack_tokens + 1
                if args.preservation_tokens is None
                else args.preservation_tokens
            ),
            "attack_tokens": min(
                args.attack_tokens, len(clean_tokens[row["id"]]) - attack_start
            ),
            "attack_start": attack_start,
            "rgb_linf_255": float(raw_delta.abs().max() * 255.0),
            "rgb_psnr_db": (-10.0 * math.log10(mse)) if mse > 0 else math.inf,
            "seed": args.seed,
            "space": "Qwen normalized resized visual patches",
        }
        atomic_torch_save(
            output,
            {
                "row_id": row["id"],
                "pixel_values": adversarial.detach().cpu(),
                "image_grid_thw": prompt_inputs["image_grid_thw"].detach().cpu(),
                "metadata": metadata,
                "trace": trace,
                "clean_target_token_ids": clean_tokens[row["id"]],
            },
        )
        completed += 1
        atomic_json(
            args.progress_json,
            {
                "status": "running",
                "completed": completed,
                "requested": len(rows),
                "last_id": row["id"],
                "updated_unix": time.time(),
            },
        )
        print(f"completed attack {completed}/{len(rows)} id={row['id']}", flush=True)

    atomic_json(
        args.progress_json,
        {
            "status": "complete",
            "completed": completed,
            "requested": len(rows),
            "method": args.method,
            "objective": args.objective,
            "epsilon_255": args.epsilon,
            "steps": args.steps,
            "preservation_weight": args.preservation_weight,
            "preservation_objective": args.preservation_objective,
            "preservation_tokens": (
                args.attack_tokens + 1
                if args.preservation_tokens is None
                else args.preservation_tokens
            ),
            "updated_unix": time.time(),
        },
    )
    print(f"visual attack complete output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
