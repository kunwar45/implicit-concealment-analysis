#!/bin/bash
# gpt-oss-20b interp v2, end to end on one H100:
#   1. serve gpt-oss, resample "confirm my order" (exact prompts, N samples each, good-news arm)
#   2. serve the judge, judge the run
#   3. teacher-force the transcripts, capture activations through the reasoning
#   4. probes vs text baseline
#   sbatch --account=def-s2ganapa_gpu scratch/interp_gptoss/interp.sbatch
#   SMOKE=1 sbatch ...   (5 scenarios, prints full model inputs/outputs)
#SBATCH --job-name=obf-interp
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
set -uo pipefail
module load python/3.12 cuda/12.9 arrow/24.0.0 opencv
source ~/scratch/obf_venv/bin/activate
cd ~/scratch/obfuscation-prompting
export HF_HOME=~/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
MODEL=openai/gpt-oss-20b; JUDGE=${JUDGE:-Qwen/Qwen2.5-32B-Instruct}
SMOKE=${SMOKE:-0}; SAMPLES=${SAMPLES:-4}
if [ "$SMOKE" = 1 ]; then TAG=smoke; GEN=(--limit 5 --show 2); else TAG=full; GEN=(); fi
OUT=results/interp_gptoss/resample_$TAG; ACT=activations/interp_gptoss; mkdir -p $ACT
PORT=$((20000 + SLURM_JOB_ID % 20000))
wait_ready() {
  for i in $(seq 1 240); do
    curl -sf localhost:$PORT/v1/models 2>/dev/null | grep -q "\"$3\"" && { echo "server ready after $((i*10))s"; return 0; }
    kill -0 $1 2>/dev/null || { echo "server died:"; tail -40 $2; return 1; }
    sleep 10
  done; echo "server NOT ready"; return 1
}
echo "### 1. generate ($TAG)"
vllm serve $MODEL --port $PORT --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.90 \
  --enforce-eager --max-num-seqs 128 > vllm-interp-$SLURM_JOB_ID.log 2>&1 &
PID=$!; wait_ready $PID vllm-interp-$SLURM_JOB_ID.log $MODEL || exit 1
export VLLM_BASE_URL=http://localhost:$PORT/v1
python -m scratch.interp_gptoss.generate --samples $SAMPLES "${GEN[@]}" --output-dir $OUT || exit 1
kill $PID; wait $PID 2>/dev/null; sleep 5
RUN=$(ls -t $OUT/run_*.json | grep -v _judged | head -1)
echo "### 2. judge $RUN"
vllm serve "$JUDGE" --port $PORT --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --enforce-eager > vllm-judge-$SLURM_JOB_ID.log 2>&1 &
PID=$!; wait_ready $PID vllm-judge-$SLURM_JOB_ID.log "$JUDGE" || exit 1
python -m scripts.judge_results "$RUN" --monitor-provider vllm --monitor-model "$JUDGE" \
  --workers 32 --judge-max-tokens 500 || exit 1
kill $PID; wait $PID 2>/dev/null; sleep 5
echo "### 3. capture"
python -m scratch.interp_gptoss.capture "${RUN%.json}_judged.json" --show 1 --out $ACT/resample_$TAG.npz || exit 1
[ "$SMOKE" = 1 ] && { echo "### done (smoke: probes skipped)"; exit 0; }
echo "### 4. probes"
python -m scratch.interp_gptoss.probe $ACT/resample_$TAG.npz --out results/interp_gptoss/v2 || exit 1
echo "### done"
