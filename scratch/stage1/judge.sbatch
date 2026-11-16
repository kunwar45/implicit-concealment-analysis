#!/bin/bash
# Interim on-cluster judge (no API keys yet): serve a local judge model with vLLM and
# run the post-hoc judge pass over one results file.
#   RESULTS=results/stage1/run_X.json sbatch --account=def-s2ganapa_gpu scratch/stage1/judge.sbatch
#SBATCH --job-name=obf-judge
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out

set -uo pipefail
module load python/3.12 cuda/12.9 arrow/24.0.0 opencv
source ~/scratch/obf_venv/bin/activate
cd ~/scratch/obfuscation-prompting
export HF_HOME=~/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

JUDGE=${JUDGE:-Qwen/Qwen2.5-32B-Instruct}
: "${RESULTS:?set RESULTS=<results json>}"
PORT=$((20000 + SLURM_JOB_ID % 20000))  # per-job port: GPU nodes are shared
vllm serve "$JUDGE" --port $PORT --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.90 --enforce-eager > vllm-judge-$SLURM_JOB_ID.log 2>&1 &
VLLM_PID=$!
for i in $(seq 1 240); do
  curl -sf localhost:$PORT/v1/models 2>/dev/null | grep -q "\"$JUDGE\"" && { echo "judge ready after $((i*10))s"; break; }
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died:"; tail -40 vllm-judge-$SLURM_JOB_ID.log; exit 1; }
  sleep 10
done
export VLLM_BASE_URL=http://localhost:$PORT/v1
python -m scripts.judge_results "$RESULTS" --monitor-provider vllm --monitor-model "$JUDGE" \
  --workers 32 --judge-max-tokens 500 ${FORCE:+--force}
STATUS=$?
kill $VLLM_PID
echo "### done (exit $STATUS)"
