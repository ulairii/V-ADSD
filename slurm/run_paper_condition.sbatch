#!/usr/bin/env bash
#SBATCH --job-name=vadsd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --signal=B:TERM@180
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Examples:
#   CONDITION=clean sbatch slurm/run_paper_condition.sbatch
#   CONDITION=pgd-lambda1 sbatch slurm/run_paper_condition.sbatch
#   CONDITION=pgd-lambda4 sbatch slurm/run_paper_condition.sbatch
#   CONDITION=pgd-unconstrained sbatch slurm/run_paper_condition.sbatch
#   CONDITION=fgsm-lambda1 sbatch slurm/run_paper_condition.sbatch
#   CONDITION=random-sign sbatch slurm/run_paper_condition.sbatch

set -euo pipefail

readonly REPO_ROOT="${REPO_ROOT:-$PWD}"
readonly WORK_ROOT="${VADSD_WORKDIR:?set VADSD_WORKDIR to shared large storage}"
readonly NUSCENES_ROOT="${NUSCENES_ROOT:?set NUSCENES_ROOT}"
readonly TARGET_MODEL="${TARGET_MODEL:?set TARGET_MODEL}"
readonly DRAFT_MODEL="${DRAFT_MODEL:?set DRAFT_MODEL}"
readonly CONDITION="${CONDITION:?set CONDITION}"
readonly PYTHON_BIN="${PYTHON_BIN:-python}"
readonly MANIFEST="${MANIFEST:-$REPO_ROOT/manifests/test.jsonl}"
readonly RUN_ROOT="$WORK_ROOT/runs/$CONDITION"
readonly CLEAN_ROOT="$WORK_ROOT/runs/clean"

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "must run inside Slurm" >&2; exit 2; }
[[ -f "$TARGET_MODEL/config.json" ]] || { echo "missing target: $TARGET_MODEL" >&2; exit 2; }
[[ -f "$DRAFT_MODEL/config.json" ]] || { echo "missing draft: $DRAFT_MODEL" >&2; exit 2; }
[[ -f "$MANIFEST" ]] || { echo "missing manifest: $MANIFEST" >&2; exit 2; }
[[ "$RUN_ROOT" == "$WORK_ROOT"/* ]] || { echo "unsafe output path" >&2; exit 2; }

mkdir -p "$RUN_ROOT" "$WORK_ROOT/cache" "$WORK_ROOT/tmp/$SLURM_JOB_ID" "$REPO_ROOT/logs"
export TMPDIR="$WORK_ROOT/tmp/$SLURM_JOB_ID"
export HF_HOME="$WORK_ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$WORK_ROOT/cache/hf_datasets"
export TORCH_HOME="$WORK_ROOT/cache/torch"
export TRITON_CACHE_DIR="$WORK_ROOT/cache/triton"
export XDG_CACHE_HOME="$WORK_ROOT/cache/xdg"
export PYTHONPATH="$REPO_ROOT/external/ViSpec:$REPO_ROOT/scripts/eval:$REPO_ROOT/scripts/data:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

run_eval() {
  local tensor_dir="${1:-}"
  local tensor_args=()
  [[ -z "$tensor_dir" ]] || tensor_args=(--adversarial-input-dir "$tensor_dir")
  "$PYTHON_BIN" "$REPO_ROOT/scripts/eval/run_drivelm_vispec.py" \
    --manifest "$MANIFEST" --image-root "$NUSCENES_ROOT" \
    --output "$RUN_ROOT/results.json" --summary-output "$RUN_ROOT/summary.json" \
    --progress-json "$RUN_ROOT/eval-progress.json" \
    --base-model "$TARGET_MODEL" --spec-model "$DRAFT_MODEL" \
    --mode paired --max-new-tokens 64 --total-token 30 --depth 3 --top-k 8 --num-q 2 \
    --max-visual-tokens-per-image 256 --torch-dtype float16 --seed 0 \
    --require-token-parity "${tensor_args[@]}"
}

if [[ "$CONDITION" == clean ]]; then
  run_eval
  exit 0
fi

[[ -f "$CLEAN_ROOT/results.json" ]] || {
  echo "run CONDITION=clean first: $CLEAN_ROOT/results.json is missing" >&2
  exit 2
}

if [[ "$CONDITION" == random-sign ]]; then
  readonly SOURCE_TENSORS="$WORK_ROOT/runs/pgd-lambda1/tensors"
  readonly RANDOM_TENSORS="$RUN_ROOT/tensors"
  [[ -d "$SOURCE_TENSORS" ]] || {
    echo "run CONDITION=pgd-lambda1 first: $SOURCE_TENSORS is missing" >&2
    exit 2
  }
  mkdir -p "$RANDOM_TENSORS"
  "$PYTHON_BIN" "$REPO_ROOT/scripts/eval/generate_matched_random_control.py" \
    --manifest "$MANIFEST" --image-root "$NUSCENES_ROOT" \
    --attack-dir "$SOURCE_TENSORS" --output-dir "$RANDOM_TENSORS" \
    --base-model "$TARGET_MODEL" --max-visual-tokens-per-image 256 --seed 2026
  run_eval "$RANDOM_TENSORS"
  exit 0
fi

method=pgd
steps=20
step_size=2
weight=1
attack_start=11
attack_tokens=24
preservation_tokens=13
case "$CONDITION" in
  pgd-lambda1) ;;
  pgd-lambda4) weight=4 ;;
  pgd-unconstrained)
    weight=0; attack_start=1; attack_tokens=24; preservation_tokens=0
    ;;
  fgsm-lambda1)
    method=fgsm; steps=1; step_size=32
    ;;
  *) echo "unsupported CONDITION=$CONDITION" >&2; exit 2 ;;
esac

readonly TENSOR_DIR="$RUN_ROOT/tensors"
mkdir -p "$TENSOR_DIR"
"$PYTHON_BIN" "$REPO_ROOT/scripts/eval/optimize_visual_attack.py" \
  --manifest "$MANIFEST" --image-root "$NUSCENES_ROOT" \
  --clean-results "$CLEAN_ROOT/results.json" \
  --output-dir "$TENSOR_DIR" --progress-json "$RUN_ROOT/attack-progress.json" \
  --base-model "$TARGET_MODEL" --spec-model "$DRAFT_MODEL" \
  --method "$method" --objective draft_ce --epsilon 32 --steps "$steps" \
  --step-size "$step_size" --preservation-weight "$weight" \
  --preservation-objective mean_ce --attack-start "$attack_start" \
  --attack-tokens "$attack_tokens" --preservation-tokens "$preservation_tokens" \
  --max-visual-tokens-per-image 256 --torch-dtype bfloat16 --seed 0

run_eval "$TENSOR_DIR"
