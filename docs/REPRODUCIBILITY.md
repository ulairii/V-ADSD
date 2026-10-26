# Reproducibility guide

## Evidence levels

This artifact supports two distinct checks:

1. **Artifact integrity (CPU-only):** validates split identity, scene
   separation, path portability, protocol fields, and every aggregate number
   printed in the paper table.
2. **Experimental reproduction (GPU):** reconstructs the target/draft pair,
   regenerates perturbations, and reruns paired target/ViSpec decoding.

The first check does not independently reproduce the GPU measurements. The
final jobs' raw per-sample files were stored outside the paper repository and
are not bundled here. Therefore a new GPU run is required to independently
verify the aggregate values.

## Data

Use nuScenes v1.0-trainval and the public DriveLM v1.1 annotation. The released
manifests contain six paths per question in this order:

`CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`, `CAM_BACK`,
`CAM_BACK_LEFT`, `CAM_BACK_RIGHT`.

Paths start at `samples/` and are resolved using `--image-root`. The 10
calibration scenes and 21 test scenes do not overlap. The frozen test set was
not used to choose epsilon, loss weights, or PGD steps.

The annotation release used by the original run was labeled v1.1, but its
content hash was not recorded at download time. This is explicitly marked as
missing provenance in `configs/paper_protocol.json`; the artifact does not
invent a revision identifier.

## Target model

The base is `Qwen/Qwen2.5-VL-7B-Instruct` at Hugging Face revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`. `prepare_target_sft.py` builds
scene-disjoint training/validation JSONL files from DriveLM. The target can be
trained and merged with:

```bash
export PYTHONPATH="$PWD/scripts/data:$PWD/scripts/eval:$PWD/external/ViSpec"

python scripts/training/prepare_target_sft.py \
  --annotation /path/to/drivelm_train_v1.1.json \
  --nuscenes-root "$NUSCENES_ROOT" \
  --exclude-manifest manifests/test.jsonl \
  --output-dir "$VADSD_WORKDIR/target-data" \
  --protocol challenge

python scripts/training/train_target_lora.py \
  --base-model /path/to/Qwen2.5-VL-7B-Instruct \
  --train-manifest "$VADSD_WORKDIR/target-data/train.jsonl" \
  --validation-manifest "$VADSD_WORKDIR/target-data/validation.jsonl" \
  --output-dir "$VADSD_WORKDIR/target-lora" \
  --epochs 1 --learning-rate 1e-4 --gradient-accumulation 8 \
  --save-every-steps 250 --max-visual-tokens-per-image 256 \
  --max-answer-tokens 400 --seed 1234 --export-merged
```

Training runs on one A100 GPU for one epoch (3,431 optimizer steps) and writes
the merged target to `$VADSD_WORKDIR/target-lora/merged_target`.

## Draft adaptation

ViSpec uses a small auxiliary draft, not a second full 7B VLM. The draft is
adapted using target-generated features from DriveLM prompts
whose scenes are excluded from evaluation.

```bash
python scripts/training/prepare_draft_adaptation.py \
  --annotation /path/to/drivelm_train_v1.1.json \
  --nuscenes-root "$NUSCENES_ROOT" \
  --exclude-manifest manifests/test.jsonl \
  --manifest "$VADSD_WORKDIR/draft-data/all.jsonl" \
  --summary "$VADSD_WORKDIR/draft-data/summary.json" --count 256

python scripts/training/generate_draft_features.py --help
python scripts/training/train_draft.py --help
```

The original final draft record contains 243 training samples, 13 validation
samples, three epochs, and final step 729. It stores one 3584-wide language
layer plus the ViSpec image adaptor and shares the target vocabulary/hidden
width expected by the ViSpec verifier.

## Evaluation sequence

1. Run `run_drivelm_vispec.py` on clean inputs with `--mode paired` and
   `--require-token-parity`.
2. Optimize tensors with `optimize_visual_attack.py`, using the clean target
   tokens from step 1.
3. Evaluate each tensor directory with the same paired runner and hardware.
4. Produce the matched-random tensor directory from the λ=1 PGD directory with
   `generate_matched_random_control.py`.
5. Compare clean and attacked target responses, audit any changed action, and
   report latency only after target/ViSpec equality passes.

The generic Slurm launcher implements steps 1–3. Separate jobs should be used
for clean, λ=0, λ=1, λ=4, FGSM, and random-sign conditions so every condition
has its own resumable output directory.

## Expected primary check

The λ=1 run should reproduce the qualitative mechanism: fewer generated tokens
per target verification call, more autoregressive decode time, unchanged
speculative-versus-target token output, and nearly unchanged target response.
Wall-clock values are hardware/software sensitive; compare all conditions on
the same GPU type and package environment. The paper's exact aggregate target
is recorded in `results/paper_results.json`.
