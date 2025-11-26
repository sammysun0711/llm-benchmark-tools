#!/bin/bash

# Model path
MODEL_PATH="/training_models/DeepSeek-R1-0528"

# Model name
MODEL_NAME="DeepSeek-R1-0528"

# Output directory
OUT_DIR="/workdir/DeepSeek-R1-0528-logs"
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
SGLANG_USE_AITER=1 python3 -m sglang.launch_server \
    	--model-path $MODEL_PATH  \
	--host localhost  \
     	--port 10000  \
     	--tensor-parallel-size 8 \
    	--trust-remote-code  \
     	--chunked-prefill-size 196608 \
    	--mem-fraction-static 0.8  \
      	--disable-radix-cache \
    	--num-continuous-decode-steps 4 \
    	--max-prefill-tokens 196608 \
	--enable-torch-compile \
    	--cuda-graph-max-bs 128 &
SERVER_PID=$!
trap 'echo "Stopping server after benchmark finished..."; kill $SERVER_PID 2>/dev/null' EXIT

# Sleep 600s
echo "Sleep 600s waiting for server to launch ..."
sleep 600

# Loop over combinations
for idx in "${!isl_list[@]}"; do
    isl=${isl_list[$idx]}
    osl=${osl_list[$idx]}

    for conc in "${conc_list[@]}"; do
        # Filenames
        result_filename="${MODEL_NAME}_isl${isl}_osl${osl}_c${conc}_sglang.json"
        log_filename="${MODEL_NAME}_isl${isl}_osl${osl}_c${conc}_sglang.log"

        echo ">>> Running benchmark: ISL=${isl}, OSL=${osl}, Concurrency=${conc}"
        # if [ $conc -ge 32 ]; then
        # NUM_PROMPT=$(( 2*conc))	
        # else
        # 	NUM_PROMPT=32
        # fi
        python3 -m sglang.bench_serving \
		--host localhost \
		--port 10000 \
		--model "$MODEL_PATH" \
		--dataset-name "$DATASET" \
		--random-input "$isl" \
		--random-output "$osl" \
		--random-range-ratio 1.0 \
		--max-concurrency "$conc" \
		--num-prompt "$NUM_PROMPT" \
		--output-file "$OUT_DIR/$result_filename" \
	        2>&1 | tee "$OUT_DIR/$log_filename"

        echo ">>> Finished benchmark: ISL=${isl}, OSL=${osl}, Concurrency=${conc}"
        echo
    done
done
