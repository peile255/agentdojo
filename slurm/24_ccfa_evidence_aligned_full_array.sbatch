#!/bin/bash
#SBATCH --job-name=ccfa-aligned-full
#SBATCH --partition=aoraki_gpu_L40
#SBATCH --gres=gpu:L40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-3%2
#SBATCH --output=ccfa_v3/logs/aligned-full-%A_%a.out
#SBATCH --error=ccfa_v3/logs/aligned-full-%A_%a.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-${PROJECT_DIR}}
cd "$PROJECT_DIR"
module load python/3.10.13-7ad3v37
source .venv/bin/activate

OUTPUT_DIR="ccfa_v3/raw/aligned_full"
mkdir -p "$OUTPUT_DIR" ccfa_v3/logs ccfa_v3/tables

STEMS=(
  clean_standard
  clean_defended
  attack_standard
  attack_defended
)

CONFIGS=(
  configs/ccfa_aligned_clean_standard_full.yaml
  configs/ccfa_aligned_clean_defended_full.yaml
  configs/ccfa_aligned_attack_standard_full.yaml
  configs/ccfa_aligned_attack_defended_full.yaml
)

INDEX=${SLURM_ARRAY_TASK_ID}
STEM=${STEMS[$INDEX]}
CONFIG=${CONFIGS[$INDEX]}
OUTPUT="$OUTPUT_DIR/${STEM}.jsonl"

if [[ -e "$OUTPUT" ]]; then
  echo "ERROR: refusing to overwrite $OUTPUT" >&2
  exit 1
fi

PORT=$((11444 + INDEX))

export OLLAMA_MODELS="${OLLAMA_MODELS}"
export OLLAMA_SIF="${OLLAMA_SIF}"
export OLLAMA_HOST="127.0.0.1:${PORT}"
export QWEN_BACKEND=openai_compatible
export QWEN_MODEL=qwen3.6:27b
export QWEN_BASE_URL="http://127.0.0.1:${PORT}/v1"

echo "date_utc=$(date -u --iso-8601=seconds)"
echo "array_job=${SLURM_ARRAY_JOB_ID}"
echo "array_task=${SLURM_ARRAY_TASK_ID}"
echo "condition=${STEM}"
echo "config=${CONFIG}"
echo "port=${PORT}"

apptainer exec --nv \
  --env OLLAMA_HOST="0.0.0.0:${PORT}" \
  --env OLLAMA_MODELS="$OLLAMA_MODELS" \
  "$OLLAMA_SIF" ollama serve \
  > "/tmp/ollama_aligned_full_${SLURM_JOB_ID}.log" 2>&1 &

OLLAMA_PID=$!
cleanup() { kill "$OLLAMA_PID" 2>/dev/null || true; }
trap cleanup EXIT

READY=0
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    READY=1
    break
  fi
  sleep 2
done

if [[ "$READY" -ne 1 ]]; then
  echo "ERROR: Ollama failed to start on port ${PORT}" >&2
  exit 1
fi

python -m src.run_collaboration_ccfa_aligned \
  --config "$CONFIG" \
  --seeds data/agentdojo_compromised_all.jsonl \
  --clean-sources ccfa_v3/native_clean/agentdojo_native_clean_paired.jsonl \
  --alignment-manifest ccfa_v3/alignment/native_evidence_alignment_manifest.jsonl \
  --output "$OUTPUT"

ROWS=$(wc -l < "$OUTPUT")
if [[ "$ROWS" -ne 44 ]]; then
  echo "ERROR: expected 44 rows, found $ROWS in $OUTPUT" >&2
  exit 1
fi

sha256sum "$OUTPUT" > "$OUTPUT.sha256"
echo "completed=${STEM} rows=${ROWS}"
