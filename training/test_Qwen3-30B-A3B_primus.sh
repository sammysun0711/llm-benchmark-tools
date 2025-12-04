#!/bin/bash

cd /workdir/Primus

# Common parameters
GBS=768
SEQ_LENGTH=4096
TP=1
ETP=1
PP=1
VPP=1
EP=8
CP=1
TRAIN_ITERS=10
SCRIPT="./examples/customer_package/run_qwen3_30b_a3b_pretrain_mi355x.sh"

# Define configurations in arrays
declare -a configs_bf16=(
    "MBS=1 MoE_Features=(0)"
    "MBS=2 MoE_Features=(0)"
    "MBS=4 MoE_Features=(0)"
    "MBS=6 MoE_Features=(0)"
    "MBS=6 MoE_Features=(0 11)"
    "MBS=6 MoE_Features=(3 11)"
    "MBS=6 MoE_Features=(0 3 11)"
    "MBS=6 MoE_Features=(0 3 4 11)"
    "MBS=6 MoE_Features=(0 3 4 5 11)"
    "MBS=6 MoE_Features=(0 3 4 5 10 11)"
)

declare -a configs_fp8=(
    "MBS=1 MoE_Features=(0)"
    "MBS=2 MoE_Features=(0)"
    "MBS=4 MoE_Features=(0)"
    "MBS=6 MoE_Features=(0)"
    "MBS=6 MoE_Features=(0 11)"
    "MBS=6 MoE_Features=(3 11)"
    "MBS=6 MoE_Features=(0 3 11)"
    "MBS=6 MoE_Features=(0 3 4 11)"
    "MBS=6 MoE_Features=(0 3 4 5 11)"
    "MBS=6 MoE_Features=(0 3 4 5 10 11)"
    "MBS=6 MoE_Features=(0 1 3 4 5 10 11)"
)

log_counter=1

# Loop BF16 configs
for cfg in "${configs_bf16[@]}"; do
    echo "Running BF16 config: $cfg"
    eval "$cfg FP8=0 GBS=$GBS SEQ_LENGTH=$SEQ_LENGTH TP=$TP ETP=$ETP PP=$PP VPP=$VPP EP=$EP CP=$CP TRAIN_ITERS=$TRAIN_ITERS $SCRIPT 2>&1 | tee test${log_counter}.log"
    ((log_counter++))
done

# Loop FP8 configs
for cfg in "${configs_fp8[@]}"; do
    echo "Running FP8 config: $cfg"
    eval "$cfg FP8=True GBS=$GBS SEQ_LENGTH=$SEQ_LENGTH TP=$TP ETP=$ETP PP=$PP VPP=$VPP EP=$EP CP=$CP TRAIN_ITERS=$TRAIN_ITERS $SCRIPT 2>&1 | tee test${log_counter}.log"
    ((log_counter++))
done