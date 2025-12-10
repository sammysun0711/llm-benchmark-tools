#!/bin/bash

# Model path
MODEL_PATH="/models/Qwen3-Omni-30B-A3B-Instruct/"

# Model name
MODEL_NAME="Qwen3-Omni-30B-A3B-Instruct"

# Output directory
OUT_DIR="${MODEL_NAME}-profile-logs"
mkdir -p "$OUT_DIR"

export SGLANG_VLM_CACHE_SIZE_MB=0
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_USE_AITER=1
export SGLANG_TORCH_PROFILER_DIR="./sglang_profile_res"
export SGLANG_PROFILE_WITH_STACK=1
export SGLANG_PROFILE_RECORD_SHAPES=1

# ISL/OSL pairs
declare -a isl_list=(8000)
declare -a osl_list=(10)
declare -a img_res_x_list=(960)
declare -a img_res_y_list=(1280)
declare -a image_count_list=(20)

# Concurrency levels
declare -a conc_list=(2)

# Metrics configuration
METRIC_PERCENTILES="50,90,95,99"
METRIC_LIST="ttft,tpot,itl,e2el"
DATASET="image"
NUM_PROMPT=4
MM_ATTENTION_BACKEND="aiter_attn" #fa3 for H20

# Launch server
python3 -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --host localhost    \
    --port 9000 \
    --tensor-parallel-size 4 \
    --data-parallel-size 2 \
    --trust-remote-code \
    --chunked-prefill-size 32768 \
    --mem-fraction-static 0.85 \
    --mm-attention-backend "${MM_ATTENTION_BACKEND}" \
    --max-prefill-tokens 32768 \
    --disable-radix-cache \
    --page-size 64 \
    --cuda-graph-max-bs 8 &
SERVER_PID=$!
trap 'echo "Stopping server after benchmark finished..."; kill $SERVER_PID 2>/dev/null' EXIT

# Sleep 300s
echo "Sleep 100s waiting for server to launch ..."
sleep 100

# Run benchmark test
# Loop over combinations
for idx in "${!isl_list[@]}"; do
    isl=${isl_list[$idx]}
    osl=${osl_list[$idx]}
    img_res_x=${img_res_x_list[$idx]}
    img_res_y=${img_res_y_list[$idx]}
    image_count=${image_count_list[$idx]}

    for conc in "${conc_list[@]}"; do
        # Filenames
        result_filename="${MODEL_NAME}_isl${isl}_osl${osl}_img_res${img_res_x}x${img_res_y}_image_count${image_count}_c${conc}_sglang.json"
        log_filename="${MODEL_NAME}_isl${isl}_osl${osl}_img_res${img_res_x}x${img_res_y}_image_count${image_count}_c${conc}_sglang.log"

        echo ">>> Running benchmark: ISL=${isl}, OSL=${osl}, Concurrency=${conc}, Image Resolution=${img_res_x}x${img_res_y}, Image Count=${image_count}"
        #if [ $conc -ge 32 ]; then
        #    NUM_PROMPT=$(( 2*conc))
        #else
        #    NUM_PROMPT=32
        #fi
        python3 -m sglang.bench_serving \
                --backend sglang-oai-chat \
                --host localhost \
                --port 9000 \
                --model "$MODEL_PATH" \
                --dataset-name "$DATASET" \
                --random-input "$isl" \
                --random-output "$osl" \
                --random-range-ratio 1.0 \
                --max-concurrency "$conc" \
                --num-prompt "$NUM_PROMPT" \
                --image-count ${image_count}   \
                --image-resolution "${img_res_x}x${img_res_y}" \
                --flush-cache \
                --skip-special-tokens \
                --profile \
                --output-file "$OUT_DIR/$result_filename" \
                2>&1 | tee "$OUT_DIR/$log_filename"

        echo ">>> Finished benchmark: ISL=${isl}, OSL=${osl}, Concurrency=${conc}, Image Resolution=${img_res_x}x${img_res_y}, Image Count=${image_count} "
        echo
    done
done
