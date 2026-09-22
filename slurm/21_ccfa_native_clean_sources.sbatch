#!/bin/bash
#SBATCH --job-name=ccfa-native-clean
#SBATCH --partition=aoraki_gpu_L40
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=ccfa_v3/logs/native-clean-%j.out
#SBATCH --error=ccfa_v3/logs/native-clean-%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-${PROJECT_DIR}}
cd "$PROJECT_DIR"

module load python/3.10.13-7ad3v37
source .venv/bin/activate

mkdir -p ccfa_v3/logs
mkdir -p ccfa_v3/native_clean
mkdir -p agentdojo_runs/ccfa_native_clean

export OLLAMA_MODELS="${OLLAMA_MODELS}"
export OLLAMA_SIF="${OLLAMA_SIF}"
export OLLAMA_HOST="127.0.0.1:11444"
export LOCAL_LLM_PORT=11444

MANIFEST="ccfa_v3/native_clean/native_clean_manifest.tsv"
PAIR_MAP="ccfa_v3/native_clean/native_clean_pair_map.json"
LOGDIR="agentdojo_runs/ccfa_native_clean"
VALIDATION="ccfa_v3/logs/native-clean-validation-${SLURM_JOB_ID}.txt"

echo "========================================================================"
echo "CCFA NATIVE CLEAN AGENTDOJO SOURCE COLLECTION"
echo "========================================================================"
echo "job=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "date_utc=$(date -u --iso-8601=seconds)"
echo "project=${PROJECT_DIR}"
echo "python=$(python --version 2>&1)"
python - <<'PY'
from importlib.metadata import version
print("agentdojo=" + version("agentdojo"))
PY

python ccfa_v3/scripts/build_native_clean_manifest.py \
  --seeds data/agentdojo_compromised_all.jsonl \
  --manifest "$MANIFEST" \
  --pair-map "$PAIR_MAP"

TASK_COUNT=$(wc -l < "$MANIFEST")
if [[ "$TASK_COUNT" -ne 9 ]]; then
    echo "ERROR: expected 9 unique clean tasks, found ${TASK_COUNT}" >&2
    exit 1
fi

nvidia-smi \
  --query-gpu=name,memory.total,memory.free \
  --format=csv,noheader

apptainer exec --nv \
  --env OLLAMA_HOST=0.0.0.0:11444 \
  --env OLLAMA_MODELS="$OLLAMA_MODELS" \
  "$OLLAMA_SIF" \
  ollama serve \
  > "/tmp/ollama_native_clean_${SLURM_JOB_ID}.log" 2>&1 &

OLLAMA_PID=$!

cleanup() {
    kill "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for Ollama..."
OLLAMA_READY=0
for _ in $(seq 1 90); do
    if curl -sf http://127.0.0.1:11444/v1/models >/dev/null; then
        OLLAMA_READY=1
        break
    fi
    sleep 2
done

if [[ "$OLLAMA_READY" -ne 1 ]]; then
    echo "ERROR: Ollama did not become ready" >&2
    exit 1
fi
echo "Ollama ready."

COUNT=0
while IFS=$'\t' read -r SUITE USER_TASK; do
    if [[ -z "$SUITE" || -z "$USER_TASK" ]]; then
        continue
    fi
    COUNT=$((COUNT + 1))
    echo
    echo "========================================================================"
    echo "CLEAN CASE ${COUNT}/${TASK_COUNT}"
    echo "suite=${SUITE}"
    echo "user_task=${USER_TASK}"
    echo "attack=omitted (native no-injection benchmark)"
    echo "========================================================================"

    python -m agentdojo.scripts.benchmark \
      --model LOCAL \
      --model-id qwen3.6:27b \
      --benchmark-version v1.2.2 \
      --suite "$SUITE" \
      --user-task "$USER_TASK" \
      --logdir "$LOGDIR"
done < "$MANIFEST"

python ccfa_v3/scripts/validate_native_clean.py \
  --logdir "$LOGDIR" \
  --manifest "$MANIFEST" \
  --seeds data/agentdojo_compromised_all.jsonl \
  | tee "$VALIDATION"

echo "========================================================================"
echo "NATIVE CLEAN SOURCE COLLECTION COMPLETED"
echo "========================================================================"
