#!/bin/bash

# Change to your work directory
cd /workdir/Megatron-LM

# Micro batch size values
mbs_list=(1 2 4 8 16 24 30)

# TE_FP8 options
fp8_list=(0 1)

# Starting test counter
test_id=1

# Loop over TE_FP8 values
for fp8 in "${fp8_list[@]}"; do
    # Loop over micro batch sizes
    for mbs in "${mbs_list[@]}"; do

        # Decide BS depending on mbs
        case $mbs in
            24) BS=192 ;;
            30) BS=240 ;;
            *)  BS=128 ;;
        esac

        # Set precision label
        if [ "$fp8" -eq 0 ]; then
            prec_label="bf16"
        else
            prec_label="fp8"
        fi

        # Log file name
        log_file="test${test_id}_${prec_label}_mbs${mbs}.log"

        echo "Running test ${test_id}: TE_FP8=${fp8}, MBS=${mbs}, BS=${BS} -> ${log_file}"

        # Run the training command
        bash examples/qwen/train_qwen3.sh \
            FSDP=1 CP=1 PP=1 TP=1 \
            MBS=${mbs} BS=${BS} TE_FP8=${fp8} \
            MODEL_SIZE=32 SEQ_LENGTH=4096 TOTAL_ITERS=10 MOCK_DATA=1 \
            RECOMPUTE_ACTIVATIONS=full CKPT_FORMAT=torch_dist \
            2>&1 | tee "${log_file}"

        # Increment test ID
        ((test_id++))
    done
done