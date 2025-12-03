### Qwen3-32B pretrain with Megatron-LM
1. Launch docker image
```bash
podman run -it  --name megatron_training_env_qwen3_pretrain --device /dev/dri --device /dev/kfd --device /dev/infiniband  --device=/dev/infiniband/rdma_cm --network host --ipc host  --cap-add=SYS_ADMIN --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged --group-add keep-groups  -v /shared/amdgpu/home/share/models:/models -v /shared/amdgpu/home/share/training_models:/training_models -v /shared/data:/shared/data -v /shared:/shared  -v $HOME:/workdir --workdir /workdir -v $HOME/.ssh:/root/.ssh  docker://rocm/megatron-lm:v25.9_gfx950
```

2. Install python dependency

- Install Megatron-LM with qwen3 changes.
```bash
pip uninstall megatron-core
git clone https://github.com/LuweiZhou2025/Megatron-LM.git && cd Megatron-LM && git checkout -b qwen3 origin/luwei/qwen3
pip install -e .
```

3. Run the script to pertain Qwen3-32B with mock data. Change the argument as needed.
```bash
bash examples/qwen/train_qwen3.sh FSDP=1 CP=1 PP=1 TP=1 MBS=24 BS=192 TE_FP8=1 MODEL_SIZE=32 SEQ_LENGTH=4096 TOTAL_ITERS=10  MOCK_DATA=1
```

### Qwen3-32B-A3B Pretrain with Primus
1. Launch docker image
```bash
podman run -it  --name primus-gfx950-ainic-qwen3-32b-a3b --device /dev/dri --device /dev/kfd --device /dev/infiniband  --device=/dev/infiniband/rdma_cm --network host --ipc host  --cap-add=SYS_ADMIN --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged --group-add keep-groups  -v /shared/amdgpu/home/share/models:/models -v /shared/amdgpu/home/share/training_models:/training_models -v /shared/data:/shared/data -v /shared:/shared  -v $HOME:/workdir --workdir /workdir -v $HOME/.ssh:/root/.ssh  docker://tasimage/primus:pr-316-gfx950-ainic
```

2. Run Qwen3-32B-A3B training scripts
```bash
git clone https://github.com/AMD-AGI/Primus.git -b dev/tas/moe_package_v2
cd Primus && git submodule update --init --recursive
MoE_Features=(0 3 4 5 10 11) MBS=6 GBS=768 SEQ_LENGTH=4096 TP=1 ETP=1 PP=1 VPP=1 EP=8 CP=1 FP8=True TRAIN_ITERS=10  ./examples/customer_package/run_qwen3_30b_a3b_pretrain_mi355x.sh
```

### Qwen3-32B-A3B SFT with LLaMA-Factory
1. Launch docker image
```bash
podman run -it  --name megatron_training_env_qwen3_sft --device /dev/dri --device /dev/kfd --device /dev/infiniband  --device=/dev/infiniband/rdma_cm --network host --ipc host  --cap-add=SYS_ADMIN --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged --group-add keep-groups  -v /shared/amdgpu/home/share/models:/models -v /shared/amdgpu/home/share/training_models:/training_models -v /shared/data:/shared/data -v /shared:/shared  -v $HOME:/workdir --workdir /workdir -v $HOME/.ssh:/root/.ssh  docker://rocm/megatron-lm:v25.9_gfx950
```

2. Install python dependency

- Install Megatron Core adapter
```bash
pip install "git+https://github.com/alibaba/roll.git#subdirectory=mcore_adapter"
```

- Install latest Megatron-LM
```bash
pip uninstall megatron-core
git clone https://github.com/ROCm/Megatron-LM && cd Megatron-LM
pip install -e .
```

- Install LLaMA-Factory
```bash
git clone https://github.com/sammysun0711/LLaMA-Factory -b qwen3_sft && cd LLaMA-Factory
pip install -r requirements.txt
pip install -e ".[torch,metrics]"
```

3. Run fine tuning
```bash
USE_MCA=1 TOKENIZERS_PARALLELISM=False OMP_NUM_THREADS=1 llamafactory-cli train examples/megatron/qwen3_moe_full.yaml
```

### Qwen3-VL-30B-A3B-Instruct SFT with LLaMA-Factory
1. Launch docker image
```bash
podman run -it  --name megatron_training_env_qwen3_vl_lora_sft --device /dev/dri --device /dev/kfd --device /dev/infiniband  --device=/dev/infiniband/rdma_cm --network host --ipc host  --cap-add=SYS_ADMIN --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged --group-add keep-groups  -v /shared/amdgpu/home/share/models:/models -v /shared/amdgpu/home/share/training_models:/training_models -v /shared/data:/shared/data -v /shared:/shared  -v $HOME:/workdir --workdir /workdir -v $HOME/.ssh:/root/.ssh  docker://rocm/megatron-lm:v25.9_gfx950
```

2. Install python dependency
```bash
git clone https://github.com/sammysun0711/LLaMA-Factory -b qwen3_sft && cd LLaMA-Factory
pip install -r requirements.txt
pip install transformers==4.57.1
pip install -e ".[torch,metrics]"
```

3. Run fine tuning
```bash
TOKENIZERS_PARALLELISM=False OMP_NUM_THREADS=1 llamafactory-cli train examples/train_lora/qwen3_vl_lora_sft.yaml
```

### DeepSeek v2 lite with Primus
1. Launch docker image
```bash
podman run -it  --name primus-gfx950-ainic-dsv2-lite --device /dev/dri --device /dev/kfd --device /dev/infiniband  --device=/dev/infiniband/rdma_cm --network host --ipc host  --cap-add=SYS_ADMIN --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged --group-add keep-groups  -v /shared/amdgpu/home/share/models:/models -v /shared/amdgpu/home/share/training_models:/training_models -v /shared/data:/shared/data -v /shared:/shared  -v $HOME:/workdir --workdir /workdir -v $HOME/.ssh:/root/.ssh  docker://tasimage/primus:pr-316-gfx950-ainic
```

2. Run deepseek v2 lite training scripts
```bash
git clone https://github.com/AMD-AGI/Primus.git
cd Primus && git checkout 7c7fc54 && git submodule update --init --recursive
./examples/moe_package/run_deepseek_v2_lite_pretrain_mi355x.sh
```

### DeepSeek v3 (3 layers) with Megatron-LM
1. Launch docker image
```bash
podman run -it  --name megatron_training_env_dsv3 --device /dev/dri --device /dev/kfd --device /dev/infiniband  --device=/dev/infiniband/rdma_cm --network host --ipc host  --cap-add=SYS_ADMIN --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged --group-add keep-groups  -v /shared/amdgpu/home/share/models:/models -v /shared/amdgpu/home/share/training_models:/training_models -v /shared/data:/shared/data -v /shared:/shared  -v $HOME:/workdir --workdir /workdir -v $HOME/.ssh:/root/.ssh  docker://rocm/megatron-lm:v25.9_gfx950
```

2. Run deepseek v3 training scripts
```bash
./test_train_megatron_lm_deepseekv3.sh
```
