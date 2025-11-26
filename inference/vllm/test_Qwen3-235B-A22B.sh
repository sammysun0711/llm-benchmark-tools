#!/bin/bash

# Model path
MODEL_PATH="/models/Qwen3-235B-A22B"

# Model name 
MODEL_NAME="Qwen3-235B-A22B"

# Output directory
OUT_DIR="/workdir/Qwen3-235B-A22B-logs-1114"
mkdir -p "$OUT_DIR"

# ISL/OSL pairs
declare -a isl_list=(256 256 1024 1024 2048 4096)
declare -a osl_list=(256 1024 256 1024 2048 4096)

# Concurrency levels
declare -a conc_list=(1 2 4 8 16 32 64 128)

# Metrics configuration
METRIC_PERCENTILES="50,90,95,99"
METRIC_LIST="ttft,tpot,itl,e2el"
DATASET="random"
NUM_PROMPT=256


# Launch server
HIP_FORCE_DEV_KERNARG=1 \
VLLM_ROCM_USE_AITER=1 \
VLLM_V1_USE_PREFILL_DECODE_ATTENTION=0 \
VLLM_ROCM_USE_AITER_MHA=1 \
VLLM_USE_V1=1 \
VLLM_WORKER_MULTIPROC_METHOD="spawn" \
SAFETENSORS_FAST_GPU=1 \
VLLM_ROCM_USE_AITER_MOE=1 \
vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 8 \
    --max-num-batched-tokens 32768 \
    --no-enable-prefix-caching \
    --disable-log-requests \
    --compilation-config '{"full_cuda_graph":true}' &
SERVER_PID=$!
trap 'echo "Stopping server after benchmark finished..."; kill $SERVER_PID 2>/dev/null' EXIT

# Sleep 600s
echo "Sleep 600s waiting for server to launch ..."
sleep 600

# Run benchmark test
# Loop over combinations
for idx in "${!isl_list[@]}"; do
    isl=${isl_list[$idx]}
    osl=${osl_list[$idx]}

    for conc in "${conc_list[@]}"; do
        # Filenames
        result_filename="${MODEL_NAME}_isl${isl}_osl${osl}_c${conc}_vllm.json"
        log_filename="${MODEL_NAME}_isl${isl}_osl${osl}_c${conc}_vllm.log"

        echo ">>> Running benchmark: ISL=${isl}, OSL=${osl}, Concurrency=${conc}"
        # if [ $conc -ge 32 ]; then
        #     NUM_PROMPT=$(( 2*conc))
        # else
	    #     NUM_PROMPT=32
        # fi
        vllm bench serve \
            --backend vllm \
            --model $MODEL_PATH \
            --served-model-name $MODEL_NAME \
            --trust-remote-code \
            --port 8000 \
            --num-prompts "$NUM_PROMPT" \
            --metric_percentiles "$METRIC_PERCENTILES" \
            --percentile-metrics "$METRIC_LIST" \
            --max-concurrency "$conc" \
            --random-input-len "$isl" \
            --random-output-len "$osl" \
            --dataset-name "$DATASET" \
            --ignore-eos \
            --save-result \
            --result-dir "$OUT_DIR" \
            --result-filename "$result_filename" \
            2>&1 | tee "$OUT_DIR/$log_filename"

        echo ">>> Finished benchmark: ISL=${isl}, OSL=${osl}, Concurrency=${conc}"
        echo
    done
done
