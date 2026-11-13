#!/bin/bash
# Smoke-test Qwen models through this repo's pipeline on one fir H100.
# Weights must already be in $HF_HOME (compute nodes have no internet).
#   sbatch --account=def-s2ganapa_gpu scratch/fir_qwen_test/test_qwen.sbatch
#SBATCH --job-name=obf-qwen-test
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

set -uo pipefail
module load python/3.12 cuda/12.9 arrow/24.0.0 opencv   # opencv: vLLM wheel dependency
source ~/scratch/obf_venv/bin/activate
cd ~/scratch/obfuscation-prompting
export HF_HOME=~/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

QWEN36=Qwen/Qwen3.6-27B
QWEN36_REV=8f33b98937c6031bb49e713dee3e8e185aafd26f   # lessons repo's pinned commit
nvidia-smi --query-gpu=name,memory.total --format=csv

if [ "${SKIP_PART1:-0}" != 1 ]; then
echo "### 1. HF in-process: Qwen2.5-7B-Instruct, tag instruction"
python -m scratch.fir_qwen_test.test_qwen --provider hf --model Qwen/Qwen2.5-7B-Instruct \
  --n-scenarios 3 --max-tokens 512
fi

echo "### 2. HF in-process: Qwen3.6-27B, thinking on, no tag instruction (activations)"
python -m scratch.fir_qwen_test.test_qwen --provider hf --model "$QWEN36" --revision "$QWEN36_REV" \
  --thinking on --no-format-instruction --n-scenarios 3 --max-tokens 1536

echo "### 3. vLLM server: Qwen3.6-27B, thinking on then off"
PORT=8000
vllm serve "$QWEN36" --revision "$QWEN36_REV" --port $PORT --dtype bfloat16 \
  --max-model-len 8192 --gpu-memory-utilization 0.90 --reasoning-parser qwen3 --enforce-eager \
  > vllm-$SLURM_JOB_ID.log 2>&1 &
VLLM_PID=$!
READY=0
for i in $(seq 1 240); do   # up to 40 min
  if curl -sf localhost:$PORT/health >/dev/null; then READY=1; echo "vLLM ready after $((i*10))s"; break; fi
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died:"; tail -40 vllm-$SLURM_JOB_ID.log; exit 1; }
  sleep 10
done
[ $READY = 1 ] || { echo "vLLM NOT ready after 40 min:"; tail -20 vllm-$SLURM_JOB_ID.log; kill $VLLM_PID; exit 1; }
export VLLM_BASE_URL=http://localhost:$PORT/v1
python -m scratch.fir_qwen_test.test_qwen --provider vllm --model "$QWEN36" --thinking on \
  --no-format-instruction --n-scenarios 3 --max-tokens 1536
python -m scratch.fir_qwen_test.test_qwen --provider vllm --model "$QWEN36" --thinking off \
  --n-scenarios 3 --max-tokens 512
kill $VLLM_PID
echo "### done"
