#!/bin/bash
# Stage 1 behaviour run on one fir H100, end to end: serve the subject model with vLLM,
# run the framing sweep (saved unjudged), then swap to the interim judge and judge it.
# Thinking is always ON (Kunwar never wants thinking-off runs):
#   TH=on      -> Qwen3-family template switch (enable_thinking=True)
#   TH=default -> models that always think (QwQ, R1-distill, gpt-oss)
#
#   MODE=smoke N=8 QTYPES=B1,R1,R2,R3 sbatch --account=def-s2ganapa_gpu scratch/stage1/stage1.sbatch
#   MODEL=Qwen/QwQ-32B RPARSER=deepseek_r1 TH=default MODE=smoke N=8 QTYPES=B1,R1,R2,R3 sbatch ...
#SBATCH --job-name=obf-stage1
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

MODEL=${MODEL:-Qwen/Qwen3.6-27B}
REV=${REV:-}; [ "$MODEL" = Qwen/Qwen3.6-27B ] && REV=${REV:-8f33b98937c6031bb49e713dee3e8e185aafd26f}
RPARSER=${RPARSER:-qwen3}; TH=${TH:-on}
JUDGE=${JUDGE:-Qwen/Qwen2.5-32B-Instruct}
MODE=${MODE:-smoke}; N=${N:-8}; WORKERS=${WORKERS:-32}; MAXTOK=${MAXTOK:-8192}
CONDS=${CONDS:-all}; QTYPES=${QTYPES:-B1,R1,R2,R3}; SEED=${SEED:-42}
TAG=$(basename "$MODEL" | tr 'A-Z.' 'a-z_'); OUT=${OUT:-results/stage1/$TAG}
COND_ARG=(); [ "$CONDS" != all ] && COND_ARG=(--conditions "$CONDS")
REV_ARG=(); [ -n "$REV" ] && REV_ARG=(--revision "$REV")
PORT=$((20000 + SLURM_JOB_ID % 20000))  # per-job port: GPU nodes are shared

wait_ready() {  # $1 pid, $2 log, $3 model id that must be served
  for i in $(seq 1 240); do
    curl -sf localhost:$PORT/v1/models 2>/dev/null | grep -q "\"$3\"" && { echo "server ready after $((i*10))s"; return 0; }
    kill -0 $1 2>/dev/null || { echo "server died:"; tail -40 $2; return 1; }
    sleep 10
  done
  echo "server NOT ready"; return 1
}

echo "### subject=$MODEL rev=${REV:-latest} parser=$RPARSER thinking=$TH questions=$QTYPES"
vllm serve "$MODEL" "${REV_ARG[@]}" --port $PORT --dtype bfloat16 --max-model-len 16384 \
  --gpu-memory-utilization 0.90 --reasoning-parser "$RPARSER" --enforce-eager \
  --max-num-seqs $WORKERS > vllm-$TAG-$SLURM_JOB_ID.log 2>&1 &
PID=$!
wait_ready $PID vllm-$TAG-$SLURM_JOB_ID.log "$MODEL" || exit 1
export VLLM_BASE_URL=http://localhost:$PORT/v1

STAMP=$(date +%s)
if [ "$MODE" = smoke ]; then RUN=(--smoke-only --smoke-scenarios "$N"); else RUN=(--skip-smoke --n-scenarios "$N"); fi
python -m scripts.run_framing "${RUN[@]}" "${COND_ARG[@]}" --query-types "$QTYPES" \
  --provider vllm --model "$MODEL" "${REV_ARG[@]}" --thinking "$TH" --no-format-instruction \
  --no-judge --max-tokens "$MAXTOK" --workers "$WORKERS" --temperature 0.7 --no-plot --seed "$SEED" \
  --output-dir "$OUT" --data-dir data/stage1
kill $PID; wait $PID 2>/dev/null; sleep 5

[ "$MODEL" = "$JUDGE" ] && echo "WARNING: judge == subject"
vllm serve "$JUDGE" --port $PORT --dtype bfloat16 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --enforce-eager > vllm-judge-$SLURM_JOB_ID.log 2>&1 &
PID=$!
wait_ready $PID vllm-judge-$SLURM_JOB_ID.log "$JUDGE" || exit 1
for f in $(find "$OUT" -name 'run_*.json' ! -name '*_judged.json' -newermt "@$STAMP"); do
  echo "### judge: $f"
  python -m scripts.judge_results "$f" --monitor-provider vllm --monitor-model "$JUDGE" \
    --workers 32 --judge-max-tokens 500
done
kill $PID
echo "### done"
