#bin/bash

NUM_LAYERS=3
#NUM_LAYERS=61 # for full deepseek-v3 model
PRECISION=fp8 #bf16
MOCK_DATA=1 
#DATA_DIR="/workspace/Megatron-LM/data/deepseek-datasets"

cd /workspace/Megatron-LM
FORCE_BANLANCE=true RUN_ENV=cluster MODEL_SIZE=671B TRAIN_ITERS=50 SEQ_LEN=4096 NUM_LAYERS=$NUM_LAYERS MICRO_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=32 PR=$PRECISION TP=1 PP=1 ETP=1 EP=8 GEMM_TUNING=1 NVTE_CK_USES_BWD_V3=1 USE_GROUPED_GEMM=true MOE_USE_LEGACY_GROUPED_GEMM=true GPT_LAYER_IN_TE=true MOCK_DATA=1 bash examples/deepseek_v3/train_deepseekv3.sh 2>&1 | tee deepsseek_v3_log.txt
