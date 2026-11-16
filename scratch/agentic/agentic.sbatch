#!/bin/bash
# Agentic concealment smoke, end to end on one fir H100:
#   1. serve Qwen3.6-27B (tool calling) and run episodes (thinking on)
#   2. swap to the interim judge (Qwen2.5-32B) and judge every run file
#   N=8 sbatch --account=def-s2ganapa_gpu scratch/agentic/agentic.sbatch
#   escalation (4 customers per conversation, subset of framings):
#   N=40 OUT=results/agentic_escalation AGENT_ARGS="--escalate 4 --n-chains 6 --conditions BASE,M_inst,P_sales,P_replace,P_csat,P_manager,P_stack,P_unmonitored,EXPLICIT" \
#     sbatch --account=def-s2ganapa_gpu --time=01:30:00 scratch/agentic/agentic.sbatch
#SBATCH --job-name=obf-agentic
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -uo pipefail
module load python/3.12 cuda/12.9 arrow/24.0.0 opencv
source ~/scratch/obf_venv/bin/activate
cd ~/scratch/obfuscation-prompting
export HF_HOME=~/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

MODEL=${MODEL:-Qwen/Qwen3.6-27B}
REV=${REV:-}; [ "$MODEL" = Qwen/Qwen3.6-27B ] && REV=${REV:-8f33b98937c6031bb49e713dee3e8e185aafd26f}
REV_ARG=(); [ -n "$REV" ] && REV_ARG=(--revision "$REV")
# gpt-oss: RPARSER=openai_gptoss TPARSER=openai THINKING=default
RPARSER=${RPARSER:-qwen3}; TPARSER=${TPARSER:-qwen3_coder}; CTX=${CTX:-49152}
JUDGE=${JUDGE:-Qwen/Qwen2.5-32B-Instruct}
# thinking ON only: Kunwar never wants thinking-off runs
N=${N:-8}; THINKING=${THINKING:-on}; WORKERS=${WORKERS:-32}
OUT=${OUT:-results/agentic}; AGENT_ARGS=${AGENT_ARGS:-}
PORT=$((20000 + SLURM_JOB_ID % 20000))  # per-job port: GPU nodes are shared

wait_ready() {  # $1 pid, $2 log, $3 model id that must be served
  for i in $(seq 1 240); do
    curl -sf localhost:$PORT/v1/models 2>/dev/null | grep -q "\"$3\"" && { echo "server ready after $((i*10))s"; return 0; }
    kill -0 $1 2>/dev/null || { echo "server died:"; tail -40 $2; return 1; }
    sleep 10
  done
  echo "server NOT ready"; return 1
}

vllm serve "$MODEL" "${REV_ARG[@]}" --port $PORT --dtype bfloat16 --max-model-len $CTX \
  --gpu-memory-utilization 0.90 --reasoning-parser "$RPARSER" --enable-auto-tool-choice \
  --tool-call-parser "$TPARSER" --enforce-eager --max-num-seqs $WORKERS \
  > vllm-agentic-$SLURM_JOB_ID.log 2>&1 &
PID=$!
wait_ready $PID vllm-agentic-$SLURM_JOB_ID.log "$MODEL" || exit 1
export VLLM_BASE_URL=http://localhost:$PORT/v1

STAMP=$(date +%s)
for TH in $THINKING; do
  echo "### agent: thinking=$TH"
  python -m scratch.agentic.run_agentic --model "$MODEL" "${REV_ARG[@]}" --thinking "$TH" \
    --n-scenarios "$N" --workers "$WORKERS" --output-dir "$OUT" --data-dir data/agentic $AGENT_ARGS
done
kill $PID; wait $PID 2>/dev/null; sleep 5

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
