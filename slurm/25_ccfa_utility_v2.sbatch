#!/bin/bash
#SBATCH --job-name=ccfa-util-v2
#SBATCH --partition=aoraki_gpu_L40
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=ccfa_v3/logs/utility-v2-%j.out
#SBATCH --error=ccfa_v3/logs/utility-v2-%j.err

set -euo pipefail

cd ${PROJECT_DIR}
module load python/3.10.13-7ad3v37
source .venv/bin/activate

mkdir -p ccfa_v3/evaluation_v2 ccfa_v3/logs

PORT=11444
export OLLAMA_MODELS="${OLLAMA_MODELS}"
export OLLAMA_SIF="${OLLAMA_SIF}"

apptainer exec --nv \
  --env OLLAMA_HOST="0.0.0.0:${PORT}" \
  --env OLLAMA_MODELS="$OLLAMA_MODELS" \
  "$OLLAMA_SIF" ollama serve \
  > "ccfa_v3/logs/utility-v2-ollama-${SLURM_JOB_ID}.log" 2>&1 &
ollama_pid=$!
trap 'kill "$ollama_pid" 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

python ccfa_v3/evaluator_v2/rescore_utility_v2.py \
  --config ccfa_v3/evaluator_v2/config_utility_v2.yaml \
  --input-dir ccfa_v3/raw/aligned_full \
  --clean-pairs ccfa_v3/native_clean/agentdojo_native_clean_paired.jsonl \
  --rubrics ccfa_v3/evaluator_v2/task_rubrics.jsonl \
  --output ccfa_v3/evaluation_v2/utility_v2_results.jsonl

python ccfa_v3/evaluator_v2/validate_utility_v2.py \
  --input ccfa_v3/evaluation_v2/utility_v2_results.jsonl \
  | tee "ccfa_v3/logs/utility-v2-validation-${SLURM_JOB_ID}.txt"

sha256sum \
  ccfa_v3/raw/aligned_full/*.jsonl \
  ccfa_v3/evaluation_v2/utility_v2_results.jsonl \
  > "ccfa_v3/evaluation_v2/utility-v2-sha256-${SLURM_JOB_ID}.txt"
