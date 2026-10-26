#!/usr/bin/env python3
"""Run target-only and/or ViSpec decoding on a fixed DriveLM manifest.

This runner intentionally uses the same ViSpec-loaded target weights and the
same processed multimodal prompt for both paths.  Generated token IDs are
trimmed at the first EOS or the exact token cap before parity is evaluated.
Raw, resumable outputs stay on scratch; a compact summary may be copied into
the repository by the Slurm launcher.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import signal
import string
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import numpy as np
from PIL import Image, ImageFilter
from transformers import AutoProcessor

from vispec.model.kv_cache import initialize_past_key_values
from vispec.model.spec_model_ours import SpecModel
from vispec.model.utils import reset_tree_mode


CAMERA_DESCRIPTION = (
    "The image is a labeled 2-by-3 surround-camera montage. Top row: "
    "CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT. Bottom row: CAM_BACK_LEFT, "
    "CAM_BACK, CAM_BACK_RIGHT. Use the label and any object reference such as "
    "<c1,CAM_FRONT,x,y> to inspect the correct camera."
)
SURROUND_DESCRIPTION = (
    "Six surround-camera views are supplied in this fixed order: CAM_FRONT, "
    "CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT. "
    "Use the camera name in an object reference such as <c1,CAM_FRONT,x,y> "
    "to inspect the corresponding view."
)
SYSTEM_PROMPT = (
    "You are an autonomous-driving visual question answering assistant. "
    "Answer the user's question using only visible evidence. Give only a "
    "concise answer, without explaining your process."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--image-root",
        type=Path,
        help="Prefix for relative nuScenes image paths in the released manifest.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--progress-json", type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--spec-model", required=True)
    parser.add_argument("--mode", choices=("target", "spec", "paired"), default="paired")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--indices",
        help="comma-separated zero-based manifest indices; overrides offset/limit",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--total-token", type=int, default=30)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--num-q", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--max-visual-tokens-per-image",
        type=int,
        default=1280,
        help="Qwen processor pixel cap expressed in 28x28 visual-token units.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Checkpoint load dtype; float32 is useful for parity diagnostics.",
    )
    parser.add_argument(
        "--require-token-parity",
        action="store_true",
        help="Fail after saving outputs unless every paired greedy sequence matches.",
    )
    parser.add_argument(
        "--visual-perturbation",
        choices=(
            "none",
            "random_sign",
            "channel_bias",
            "checkerboard",
            "sinusoid",
            "gaussian_blur",
            "jpeg",
            "patch",
        ),
        default="none",
        help=(
            "Deterministic image condition applied before preprocessing; additive "
            "modes use an RGB-equivalent budget, while blur/JPEG/patch use severity."
        ),
    )
    parser.add_argument(
        "--perturb-epsilon",
        type=int,
        default=0,
        help=(
            "Integer strength: additive modes use an L-infinity budget on 0..255; "
            "physical-style modes map it to blur/JPEG/patch severity."
        ),
    )
    parser.add_argument("--perturb-phase", type=int, default=0)
    parser.add_argument("--perturb-period", type=int, default=8)
    parser.add_argument(
        "--adversarial-input-dir",
        type=Path,
        help=(
            "Optional directory of per-row preprocessed visual tensors produced by "
            "optimize_drivelm_vispec_visual_attack.py."
        ),
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def load_jsonl(
    path: Path,
    limit: int | None,
    offset: int = 0,
    indices: str | None = None,
    image_root: Path | None = None,
) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if indices:
        selected = [int(value.strip()) for value in indices.split(",")]
        if len(selected) != len(set(selected)) or any(
            index < 0 or index >= len(rows) for index in selected
        ):
            raise ValueError(f"invalid or duplicate manifest indices: {indices}")
        rows = [rows[index] for index in selected]
    else:
        rows = rows[offset:]
        rows = rows if limit is None else rows[:limit]
    if image_root is not None:
        for row in rows:
            if row.get("image_paths"):
                row["image_paths"] = [
                    str(resolve_image_path(image_root, path_value))
                    for raw_path in row["image_paths"]
                    for path_value in (Path(raw_path),)
                ]
            elif row.get("image_path"):
                path_value = Path(row["image_path"])
                row["image_path"] = str(resolve_image_path(image_root, path_value))
    return rows


def resolve_image_path(image_root: Path, path_value: Path) -> Path:
    """Resolve a released relative path from a nuScenes leaf or bundle root."""
    if path_value.is_absolute():
        return path_value
    direct = image_root / path_value
    nested = image_root / "data_full" / "nuscenes" / path_value
    if direct.is_file() or not nested.is_file():
        return direct
    return nested


def safe_row_name(row_id: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in row_id)


def apply_adversarial_inputs(
    inputs: dict[str, torch.Tensor],
    row: dict[str, Any],
    adversarial_input_dir: Path | None,
) -> dict[str, Any] | None:
    if adversarial_input_dir is None:
        return None
    artifact_path = adversarial_input_dir / f"{safe_row_name(row['id'])}.pt"
    if not artifact_path.is_file():
        raise FileNotFoundError(f"missing adversarial input artifact: {artifact_path}")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if artifact.get("row_id") != row["id"]:
        raise RuntimeError(
            f"adversarial row mismatch: expected {row['id']!r}, "
            f"found {artifact.get('row_id')!r}"
        )
    if "pixel_values" not in inputs:
        raise RuntimeError("adversarial input requested for a prompt without pixel_values")
    adversarial = artifact["pixel_values"]
    if tuple(adversarial.shape) != tuple(inputs["pixel_values"].shape):
        raise RuntimeError(
            "adversarial pixel_values shape mismatch: "
            f"expected {tuple(inputs['pixel_values'].shape)}, "
            f"found {tuple(adversarial.shape)}"
        )
    if "image_grid_thw" in artifact and not torch.equal(
        artifact["image_grid_thw"].cpu(), inputs["image_grid_thw"].cpu()
    ):
        raise RuntimeError(f"adversarial image grid mismatch for {row['id']}")
    inputs["pixel_values"] = adversarial.to(
        device=inputs["pixel_values"].device,
        dtype=inputs["pixel_values"].dtype,
    )
    return {
        key: value
        for key, value in artifact.get("metadata", {}).items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }


def perturb_image(
    image: Image.Image,
    image_path: str,
    mode: str = "none",
    epsilon: int = 0,
    seed: int = 0,
    phase: int = 0,
    period: int = 8,
) -> Image.Image:
    if mode == "none" or epsilon == 0:
        return image
    if not 0 <= epsilon <= 32:
        raise ValueError("perturb_epsilon must be in [0, 32]")
    if period < 2:
        raise ValueError("perturb_period must be at least 2")
    # These three conditions model common physical/camera effects.  Their
    # severity is intentionally kept in the same small integer control used by
    # the additive baselines, but they are not L-infinity attacks.
    if mode == "gaussian_blur":
        radius = max(0.1, epsilon / 4.0)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    if mode == "jpeg":
        quality = max(20, 100 - 10 * epsilon)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, subsampling=2)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB").copy()
    if mode == "patch":
        pixels = np.asarray(image, dtype=np.float32).copy()
        height, width, _ = pixels.shape
        patch_h = max(8, int(round(height * 0.12)))
        patch_w = max(8, int(round(width * 0.12)))
        digest = hashlib.sha256(f"patch:{seed}:{image_path}".encode()).digest()
        x0 = int.from_bytes(digest[:4], "little") % max(1, width - patch_w + 1)
        y0 = int.from_bytes(digest[4:8], "little") % max(1, height - patch_h + 1)
        color = np.asarray((255, 255, 255) if phase % 2 == 0 else (0, 0, 0), dtype=np.float32)
        alpha = min(1.0, max(0.0, epsilon / 16.0))
        pixels[y0 : y0 + patch_h, x0 : x0 + patch_w] = (
            (1.0 - alpha) * pixels[y0 : y0 + patch_h, x0 : x0 + patch_w]
            + alpha * color
        )
        return Image.fromarray(np.rint(np.clip(pixels, 0, 255)).astype(np.uint8), mode="RGB")
    pixels = np.asarray(image, dtype=np.int16)
    height, width, _ = pixels.shape
    yy, xx = np.indices((height, width))
    if mode == "random_sign":
        digest = hashlib.sha256(f"{seed}:{image_path}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        delta = rng.choice((-epsilon, epsilon), size=pixels.shape)
    elif mode == "channel_bias":
        channel_patterns = (
            (1, -1, 1),
            (-1, 1, 1),
            (1, 1, -1),
            (-1, -1, 1),
        )
        signs = np.asarray(channel_patterns[phase % len(channel_patterns)], dtype=np.int16)
        delta = epsilon * signs[None, None, :]
    elif mode == "checkerboard":
        sign = (((xx // period + yy // period + phase) % 2) * 2 - 1)[..., None]
        channel_sign = np.asarray([1, -1, 1], dtype=np.int16)[None, None, :]
        delta = epsilon * sign * channel_sign
    elif mode == "sinusoid":
        angle = 2.0 * np.pi * (xx + 2 * yy + phase) / period
        waves = np.stack(
            (np.sin(angle), np.sin(angle + 2.0 * np.pi / 3.0), np.sin(angle + 4.0 * np.pi / 3.0)),
            axis=-1,
        )
        delta = np.rint(epsilon * waves).astype(np.int16)
    else:
        raise ValueError(f"unsupported visual perturbation: {mode}")
    changed = np.clip(pixels + delta, 0, 255).astype(np.uint8)
    return Image.fromarray(changed, mode="RGB")


def build_inputs(
    processor: Any,
    row: dict[str, Any],
    visual_perturbation: str = "none",
    perturb_epsilon: int = 0,
    perturb_seed: int = 0,
    perturb_phase: int = 0,
    perturb_period: int = 8,
) -> dict[str, torch.Tensor]:
    image_paths = row.get("image_paths")
    if image_paths:
        camera_order = row.get("camera_order") or [
            f"CAMERA_{index}" for index in range(len(image_paths))
        ]
        if len(camera_order) != len(image_paths):
            raise ValueError("camera_order and image_paths must have equal lengths")
        user_content: list[dict[str, str]] = [
            {"type": "text", "text": SURROUND_DESCRIPTION}
        ]
        for camera in camera_order:
            user_content.extend(
                [
                    {"type": "text", "text": f"\n{camera}:"},
                    {"type": "image"},
                ]
            )
        reference = row.get("reference_answer", "").strip()
        if reference in {"A", "B", "C", "D"}:
            answer_instruction = "Return only the option letter (A, B, C, or D)."
        elif reference.lower().rstrip(".") in {"yes", "no"}:
            answer_instruction = "Return exactly Yes. or No."
        else:
            answer_instruction = "Return only the concise answer."
        user_content.append(
            {
                "type": "text",
                "text": f"\nQuestion: {row['question']}\n{answer_instruction}",
            }
        )
        images = []
        for image_path in image_paths:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                images.append(
                    perturb_image(
                        image,
                        image_path,
                        visual_perturbation,
                        perturb_epsilon,
                        perturb_seed,
                        perturb_phase,
                        perturb_period,
                    )
                )
    else:
        user_content = [
            {"type": "image"},
            {
                "type": "text",
                "text": f"{CAMERA_DESCRIPTION}\n\nQuestion: {row['question']}",
            },
        ]
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
            images = [
                perturb_image(
                    image,
                    row["image_path"],
                    visual_perturbation,
                    perturb_epsilon,
                    perturb_seed,
                    perturb_phase,
                    perturb_period,
                )
            ]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    return processor(images=images, text=prompt, return_tensors="pt").to("cuda:0")


def ensure_cache(model: SpecModel) -> tuple[Any, Any, Any]:
    if not hasattr(model, "past_key_values"):
        try:
            cache = initialize_past_key_values(model.base_model)
        except Exception:
            cache = initialize_past_key_values(model.base_model.language_model)
        model.past_key_values, model.past_key_values_data, model.current_length_data = cache
    model.current_length_data.zero_()
    return model.past_key_values, model.past_key_values_data, model.current_length_data


@torch.inference_mode()
def target_greedy(
    model: SpecModel,
    inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
) -> tuple[list[int], float, float]:
    model.spec_layer.reset_kv()
    past_key_values, _, _ = ensure_cache(model)
    reset_tree_mode(model)
    input_ids = inputs["input_ids"].clone()
    model_kwargs = {key: value for key, value in inputs.items() if key != "input_ids"}

    torch.cuda.synchronize()
    total_start = time.perf_counter()
    outputs = model.base_model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
        **model_kwargs,
    )
    torch.cuda.synchronize()
    decode_start = time.perf_counter()
    generated: list[int] = []
    eos_id = model.tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        token = outputs.logits[:, -1:].argmax(dim=-1)
        token_id = int(token.item())
        generated.append(token_id)
        if token_id == eos_id:
            break
        outputs = model.base_model(token, use_cache=True, past_key_values=past_key_values)
    torch.cuda.synchronize()
    end = time.perf_counter()
    return generated, end - total_start, end - decode_start


def trim_generated(token_ids: list[int], eos_id: int, cap: int) -> list[int]:
    trimmed = token_ids[:cap]
    if eos_id in trimmed:
        trimmed = trimmed[: trimmed.index(eos_id) + 1]
    return trimmed


@torch.inference_mode()
def spec_greedy(
    model: SpecModel,
    inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
) -> tuple[list[int], float, float, list[int], int]:
    input_len = int(inputs["input_ids"].shape[1])
    torch.cuda.synchronize()
    total_start = time.perf_counter()
    (
        output_ids,
        reported_new_tokens,
        verifier_index,
        acceptance,
        decode_time,
    ) = model.specgenerate(
        **inputs,
        temperature=0.0,
        max_new_tokens=max_new_tokens - 1,
        max_length=max(4096, input_len + max_new_tokens + 128),
        log=True,
        return_acceptance_len=True,
        return_decode_time=True,
    )
    torch.cuda.synchronize()
    end = time.perf_counter()
    generated = output_ids[0, input_len:].tolist()
    generated = trim_generated(generated, model.tokenizer.eos_token_id, max_new_tokens)
    return (
        generated,
        end - total_start,
        float(decode_time),
        acceptance,
        int(verifier_index) + 1,
    )


def decode(tokenizer: Any, token_ids: list[int]) -> str:
    vocabulary_size = len(tokenizer)
    safe_ids = [token if 0 <= token < vocabulary_size else 0 for token in token_ids]
    return tokenizer.decode(safe_ids, skip_special_tokens=True).strip()


def normalize_answer(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, reference: str) -> float:
    pred = normalize_answer(prediction).split()
    ref = normalize_answer(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred = normalize_answer(prediction).split()
    ref = normalize_answer(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    previous = [0] * (len(ref) + 1)
    for pred_token in pred:
        current = [0]
        for index, ref_token in enumerate(ref, start=1):
            if pred_token == ref_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    lcs = previous[-1]
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def summarize(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    summary: dict[str, Any] = {"mode": mode, "num_samples": len(rows)}
    for path in ("target", "spec"):
        available = [row[path] for row in rows if path in row]
        if not available:
            continue
        summary[path] = {
            "mean_total_time_sec": sum(item["total_time_sec"] for item in available)
            / len(available),
            "mean_decode_time_sec": sum(item["decode_time_sec"] for item in available)
            / len(available),
            "mean_generated_tokens": sum(len(item["token_ids"]) for item in available)
            / len(available),
            "normalized_exact_match": sum(
                normalize_answer(item["text"]) == normalize_answer(item["reference_answer"])
                for item in available
            )
            / len(available),
            "mean_token_f1": sum(
                token_f1(item["text"], item["reference_answer"]) for item in available
            )
            / len(available),
            "mean_rouge_l_f1": sum(
                rouge_l_f1(item["text"], item["reference_answer"]) for item in available
            )
            / len(available),
        }
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if path in row:
                by_task[row["task"]].append(row[path])
        summary[path]["by_task"] = {
            task: {
                "count": len(items),
                "normalized_exact_match": sum(
                    normalize_answer(item["text"])
                    == normalize_answer(item["reference_answer"])
                    for item in items
                )
                / len(items),
                "mean_token_f1": sum(
                    token_f1(item["text"], item["reference_answer"]) for item in items
                )
                / len(items),
            }
            for task, items in sorted(by_task.items())
        }
    paired = [row for row in rows if "target" in row and "spec" in row]
    if paired:
        exact = sum(row["target"]["token_ids"] == row["spec"]["token_ids"] for row in paired)
        spec_time = sum(row["spec"]["total_time_sec"] for row in paired)
        target_time = sum(row["target"]["total_time_sec"] for row in paired)
        acceptance_values = [
            value for row in paired for value in row["spec"].get("acceptance_length", [])
        ]
        total_spec_tokens = sum(len(row["spec"]["token_ids"]) for row in paired)
        total_verifier_steps = sum(row["spec"]["verifier_steps"] for row in paired)
        summary["paired"] = {
            "token_parity_rate": exact / len(paired),
            "num_exact": exact,
            "num_paired": len(paired),
            "speedup_total": target_time / spec_time if spec_time else None,
            "mean_raw_accept_length": (
                sum(acceptance_values) / len(acceptance_values)
                if acceptance_values
                else None
            ),
            "generated_tokens_per_verifier_step": (
                total_spec_tokens / total_verifier_steps if total_verifier_steps else None
            ),
        }
    return summary


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    torch.manual_seed(args.seed)
    manifest = load_jsonl(
        args.manifest, args.limit, args.offset, args.indices, args.image_root
    )
    progress_path = args.progress_json or args.output.with_suffix(".progress.json")
    summary_path = args.summary_output or args.output.with_suffix(".summary.json")
    existing: list[dict[str, Any]] = []
    if args.output.is_file():
        existing = json.loads(args.output.read_text())
    completed_ids = {row["id"] for row in existing}

    processor = AutoProcessor.from_pretrained(
        args.base_model,
        use_fast=True,
        min_pixels=256 * 28 * 28,
        max_pixels=args.max_visual_tokens_per_image * 28 * 28,
        local_files_only=True,
    )
    torch_dtype: str | torch.dtype = args.torch_dtype
    if args.torch_dtype != "auto":
        torch_dtype = getattr(torch, args.torch_dtype)
    model = SpecModel.from_pretrained(
        base_model_path=args.base_model,
        spec_model_path=args.spec_model,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        num_q=args.num_q,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()

    stop_requested = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        print(f"Received signal {signum}; saving after the current sample", flush=True)
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    if manifest and not existing and args.warmup > 0:
        warmup_inputs = build_inputs(
            processor,
            manifest[0],
            args.visual_perturbation,
            args.perturb_epsilon,
            args.seed,
            args.perturb_phase,
            args.perturb_period,
        )
        apply_adversarial_inputs(
            warmup_inputs, manifest[0], args.adversarial_input_dir
        )
        for _ in range(args.warmup):
            if args.mode in ("target", "paired"):
                target_greedy(model, warmup_inputs, min(args.max_new_tokens, 16))
            if args.mode in ("spec", "paired"):
                spec_greedy(model, warmup_inputs, min(args.max_new_tokens, 16))
        print(f"completed {args.warmup} warmup iteration(s)", flush=True)

    results = list(existing)
    for index, row in enumerate(manifest):
        if row["id"] in completed_ids:
            continue
        inputs = build_inputs(
            processor,
            row,
            args.visual_perturbation,
            args.perturb_epsilon,
            args.seed,
            args.perturb_phase,
            args.perturb_period,
        )
        adversarial_metadata = apply_adversarial_inputs(
            inputs, row, args.adversarial_input_dir
        )
        result: dict[str, Any] = {
            "id": row["id"],
            "task": row["task"],
            "question": row["question"],
            "tag": row.get("tag"),
            "scene_id": row.get("scene_id"),
            "frame_id": row.get("frame_id"),
            "image_path": row.get("image_path"),
            "image_paths": row.get("image_paths"),
            "visual_perturbation": {
                "mode": args.visual_perturbation,
                "epsilon_255": args.perturb_epsilon,
                "phase": args.perturb_phase,
                "period": args.perturb_period,
                "seed": args.seed,
                "adversarial_input_dir": (
                    str(args.adversarial_input_dir)
                    if args.adversarial_input_dir is not None
                    else None
                ),
                "adversarial_metadata": adversarial_metadata,
            },
        }
        if args.mode in ("target", "paired"):
            ids, total_time, decode_time = target_greedy(
                model, inputs, args.max_new_tokens
            )
            result["target"] = {
                "token_ids": ids,
                "text": decode(model.tokenizer, ids),
                "reference_answer": row["reference_answer"],
                "total_time_sec": total_time,
                "decode_time_sec": decode_time,
            }
        if args.mode in ("spec", "paired"):
            ids, total_time, decode_time, acceptance, verifier_steps = spec_greedy(
                model, inputs, args.max_new_tokens
            )
            result["spec"] = {
                "token_ids": ids,
                "text": decode(model.tokenizer, ids),
                "reference_answer": row["reference_answer"],
                "total_time_sec": total_time,
                "decode_time_sec": decode_time,
                "acceptance_length": acceptance,
                "verifier_steps": verifier_steps,
            }
        results.append(result)
        completed_ids.add(row["id"])
        print(f"completed {len(results)}/{len(manifest)} id={row['id']}", flush=True)
        if len(results) % args.save_every == 0:
            atomic_json(args.output, results)
            atomic_json(
                progress_path,
                {
                    "mode": args.mode,
                    "completed": len(results),
                    "total": len(manifest),
                    "updated_unix": time.time(),
                },
            )
        if stop_requested:
            break

    result_summary = summarize(results, args.mode)
    result_summary["data_slice"] = {"offset": args.offset, "limit": args.limit}
    result_summary["visual_perturbation"] = {
        "mode": args.visual_perturbation,
        "epsilon_255": args.perturb_epsilon,
        "phase": args.perturb_phase,
        "period": args.perturb_period,
        "seed": args.seed,
        "adversarial_input_dir": (
            str(args.adversarial_input_dir)
            if args.adversarial_input_dir is not None
            else None
        ),
    }
    atomic_json(args.output, results)
    atomic_json(summary_path, result_summary)
    atomic_json(
        progress_path,
        {
            "mode": args.mode,
            "completed": len(results),
            "total": len(manifest),
            "finished": len(results) == len(manifest),
            "updated_unix": time.time(),
        },
    )
    if stop_requested:
        raise SystemExit(143)
    if args.require_token_parity and args.mode == "paired":
        parity = result_summary.get("paired", {}).get("token_parity_rate")
        if parity != 1.0:
            raise RuntimeError(f"token parity gate failed: {parity!r}")


if __name__ == "__main__":
    main()
