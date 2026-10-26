#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly VISPEC_DIR="$REPO_ROOT/external/ViSpec"
readonly VISPEC_URL="https://github.com/KangJialiang/ViSpec.git"
readonly VISPEC_COMMIT="49a33ff0ca31e281b9dc8803fd6f3041c6c0e135"

if [[ ! -d "$VISPEC_DIR/.git" ]]; then
  mkdir -p "$(dirname "$VISPEC_DIR")"
  git clone "$VISPEC_URL" "$VISPEC_DIR"
else
  if [[ -n "$(git -C "$VISPEC_DIR" status --porcelain)" ]]; then
    all_applied=true
    for patch in \
      "$REPO_ROOT/patches/vispec_multi_image.patch" \
      "$REPO_ROOT/patches/vispec_qwen_gradient_no_cache.patch"; do
      if ! git -C "$VISPEC_DIR" apply --reverse --check "$patch" >/dev/null 2>&1; then
        all_applied=false
      fi
    done
    if [[ "$all_applied" == true ]] \
      && [[ "$(git -C "$VISPEC_DIR" rev-parse HEAD)" == "$VISPEC_COMMIT" ]]; then
      echo "ViSpec already prepared at $VISPEC_DIR ($VISPEC_COMMIT)"
      exit 0
    fi
    echo "refusing to alter a ViSpec checkout with unrelated modifications: $VISPEC_DIR" >&2
    exit 2
  fi
fi

git -C "$VISPEC_DIR" fetch --all --tags
git -C "$VISPEC_DIR" checkout --detach "$VISPEC_COMMIT"

for patch in \
  "$REPO_ROOT/patches/vispec_multi_image.patch" \
  "$REPO_ROOT/patches/vispec_qwen_gradient_no_cache.patch"; do
  if git -C "$VISPEC_DIR" apply --reverse --check "$patch" >/dev/null 2>&1; then
    echo "already applied: $(basename "$patch")"
  else
    git -C "$VISPEC_DIR" apply --check "$patch"
    git -C "$VISPEC_DIR" apply "$patch"
    echo "applied: $(basename "$patch")"
  fi
done

echo "ViSpec ready at $VISPEC_DIR ($(git -C "$VISPEC_DIR" rev-parse HEAD))"
