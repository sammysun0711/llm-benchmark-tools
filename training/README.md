## DeepSeek v3

1. Launch docker image
```bash
podman run -it  --name megatron_training_env --device /dev/dri --device /dev/kfd --device /dev/infiniband  --device=/dev/infiniband/rdma_cm --network host --ipc host  --cap-add=SYS_ADMIN --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged --group-add keep-groups  -v /shared/amdgpu/home/share/models:/models -v /shared/amdgpu/home/share/training_models:/training_models -v /shared/data:/shared/data -v /shared:/shared  -v $HOME:/workdir --workdir /workdir -v $HOME/.ssh:/root/.ssh  rocm/megatron-lm:v25.9_gfx950
```

2. Run deepseek v3 training scripts
```bash
./test_train_megatron_lm_deepseekv3.sh
```
