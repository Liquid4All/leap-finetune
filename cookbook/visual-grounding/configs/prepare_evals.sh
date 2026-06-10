#!/bin/bash
# CPU-only SLURM job that builds the RefCOCO + RefCOCO+ + RefCOCOg val
# eval jsonls from the jxu124/* HF datasets and writes one jsonl per
# benchmark.
#
# The jxu124/* datasets ship referring expressions + bbox annotations but
# only IMAGE REFERENCES (file_name), not image bytes — so this job first
# fetches the COCO 2014 train images the annotations point at (~13 GB,
# one-time; cached and skipped on reruns). Override the cache location with
# COCO_PARENT=/your/path. Total fresh-run disk ~20 GB; reruns are seconds.

#SBATCH --job-name=grounding-prep-evals
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/grounding/EVAL_PREP_OUT_%j
#SBATCH --error=logs/grounding/EVAL_PREP_ERR_%j

set -euo pipefail

mkdir -p logs/grounding

if [ -f "$HOME/.env" ]; then
    set -a; source "$HOME/.env"; set +a
fi

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1

# COCO 2014 train images (the jxu124/* annotations reference these by
# filename). Cached under ./data/coco/train2014 by default; set COCO_PARENT
# to reuse an existing COCO download elsewhere.
#
# We gate on the EXACT image count (train2014 has 82,783 jpgs), not just
# "directory non-empty" — an interrupted wget/unzip leaves a partial dir
# that would otherwise be silently accepted and yield a truncated eval set.
# A short count → wipe and re-download (atomic enough for a one-off prep).
COCO_PARENT="${COCO_PARENT:-$PWD/data/coco}"
COCO_DIR="$COCO_PARENT/train2014"
COCO_EXPECTED=82783
# Return 0 (not a find error) when the dir is absent — under `set -euo
# pipefail`, letting `find` fail on a missing path would abort the script
# before the first-run download branch can fire.
coco_count() {
    [ -d "$COCO_DIR" ] || { echo 0; return 0; }
    find "$COCO_DIR" -maxdepth 1 -name 'COCO_train2014_*.jpg' 2>/dev/null | wc -l
}
n=$(coco_count)
if [ "$n" -lt "$COCO_EXPECTED" ]; then
    echo "[evals] COCO 2014 train images incomplete at $COCO_DIR ($n/$COCO_EXPECTED)"
    echo "[evals] (re)downloading train2014.zip (~13 GB, one-time)…"
    rm -rf "$COCO_DIR"  # clear any partial extract before re-fetching
    mkdir -p "$COCO_PARENT"
    wget -q -O "$COCO_PARENT/train2014.zip" \
        http://images.cocodataset.org/zips/train2014.zip
    # The zip extracts to a train2014/ subdir, landing files at $COCO_DIR.
    unzip -q "$COCO_PARENT/train2014.zip" -d "$COCO_PARENT"
    rm -f "$COCO_PARENT/train2014.zip"
    n=$(coco_count)
    if [ "$n" -lt "$COCO_EXPECTED" ]; then
        echo "[evals] ERROR: COCO extract still incomplete ($n/$COCO_EXPECTED)." >&2
        echo "[evals] Refusing to build a truncated eval set. Check disk/network." >&2
        exit 1
    fi
fi
echo "[evals] COCO ready: $n images at $COCO_DIR"

# Full val splits (3811 + 3805 + 2573) keep the ±1 IoU-point noise floor.
uv run python cookbook/visual-grounding/prepare_evals.py \
    --output ./data/grounding_evals \
    --coco-train2014 "$COCO_DIR" \
    --limit 5000
