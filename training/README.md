### Qwen3-32B-A3B SFT
1. Launch docker image
```bash
podman run -it  --name megatron_training_env --device /dev/dri --device /dev/kfd --device /dev/infiniband  --device=/dev/infiniband/rdma_cm --network host --ipc host  --cap-add=SYS_ADMIN --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged --group-add keep-groups  -v /shared/amdgpu/home/share/models:/models -v /shared/amdgpu/home/share/training_models:/training_models -v /shared/data:/shared/data -v /shared:/shared  -v $HOME:/workdir --workdir /workdir -v $HOME/.ssh:/root/.ssh  docker://rocm/megatron-lm:v25.9_gfx950
```

2. Install python dependency

2.1. Install Megatron Core adapter
```bash
pip install "git+https://github.com/alibaba/roll.git#subdirectory=mcore_adapter"
```

2.2. Install latest Megatron-LM
```bash
pip uninstall megatron-core
git clone https://github.com/ROCm/Megatron-LM && cd Megatron-LM
pip install -e .
```

2.3. Install LLaMA-Factory
```bash
git clone https://github.com/sammysun0711/LLaMA-Factory -b qwen3_sft && cd LLaMA-Factory
pip install -r requirements.txt
pip install -e ".[torch,metrics]"
```

3. Run fine tuning
```bash
USE_MCA=1 TOKENIZERS_PARALLELISM=False OMP_NUM_THREADS=1 llamafactory-cli train examples/megatron/qwen3_moe_full.yaml
```

### Qwen3-VL-30B-A3B SFT
1. Launch docker image
```bash
podman run -it  --name megatron_training_env --device /dev/dri --device /dev/kfd --device /dev/infiniband  --device=/dev/infiniband/rdma_cm --network host --ipc host  --cap-add=SYS_ADMIN --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged --group-add keep-groups  -v /shared/amdgpu/home/share/models:/models -v /shared/amdgpu/home/share/training_models:/training_models -v /shared/data:/shared/data -v /shared:/shared  -v $HOME:/workdir --workdir /workdir -v $HOME/.ssh:/root/.ssh  docker://rocm/megatron-lm:v25.9_gfx950
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

### DeepSeek v3
1. Launch docker image
```bash
podman run -it  --name megatron_training_env --device /dev/dri --device /dev/kfd --device /dev/infiniband  --device=/dev/infiniband/rdma_cm --network host --ipc host  --cap-add=SYS_ADMIN --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged --group-add keep-groups  -v /shared/amdgpu/home/share/models:/models -v /shared/amdgpu/home/share/training_models:/training_models -v /shared/data:/shared/data -v /shared:/shared  -v $HOME:/workdir --workdir /workdir -v $HOME/.ssh:/root/.ssh  docker://rocm/megatron-lm:v25.9_gfx950
```

2. Run deepseek v3 training scripts
```bash
./test_train_megatron_lm_deepseekv3.sh
```
