## Qwen2.5-72B-Instruct
1. Launch vllm docker image
```
podman run -it --name rocm_vllm_dev_nightly_main_20251114  --device=/dev/dri --device=/dev/kfd --device=/dev/infiniband --device=/dev/infiniband/rdma_cm --privileged --network=host --ipc=host --cap-add=SYS_ADMIN --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --group-add keep-groups --ipc=host -v /shared/amdgpu/home/share/models:/models -v /shared/data:/shared/data -v /shared:/shared  -v $HOME:/workdir --workdir /workdir  docker://rocm/vllm-dev:nightly_main_20251114 bash
```

2. Benchmark Qwen2.5_72B_Instruct with vllm
```bash
./test_Qwen2.5-72B-Instruct.sh
```

3. Parse performance data and saved in csv
```python
python parse_results_vllm.py --input_dir Qwen2.5-72B-Instruct-logs/ --output_file Qwen2.5-72B-Instruct-vllm_benchmark_results.csv
```

## Qwen3-235b-A22B
1. Launch vllm docker image
```
podman run -it --name rocm_vllm_dev_nightly_main_20251114  --device=/dev/dri --device=/dev/kfd --device=/dev/infiniband --device=/dev/infiniband/rdma_cm --privileged --network=host --ipc=host --cap-add=SYS_ADMIN --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --group-add keep-groups --ipc=host -v /shared/amdgpu/home/share/models:/models -v /shared/data:/shared/data -v /shared:/shared  -v $HOME:/workdir --workdir /workdir  docker://rocm/vllm-dev:nightly_main_20251114 bash
```

2. Benchmark Qwen3-235b-A22B with vllm
```bash
./test_Qwen3-235b-A22B.sh
```

3. Parse performance data and saved in csv
```python
python parse_results_vllm.py --input_dir Qwen3-235B-A22B-logs --output_file Qwen3-235B-A22B-vllm_benchmark_results.csv
```

## Qwen3-VL-235B-A22B-Instruct
1. Launch vllm docker image
```
podman run -it --name rocm_vllm_dev_nightly_main_20251114  --device=/dev/dri --device=/dev/kfd --device=/dev/infiniband --device=/dev/infiniband/rdma_cm --privileged --network=host --ipc=host --cap-add=SYS_ADMIN --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --group-add keep-groups --ipc=host -v /shared/amdgpu/home/share/models:/models -v /shared/data:/shared/data -v /shared:/shared  -v $HOME:/workdir --workdir /workdir  docker://rocm/vllm-dev:nightly_main_20251114 bash
```

2. Benchmark Qwen3-VL-235B-A22B-Instruct with vllm
```bash
./test_Qwen3-VL-235B-A22B-Instruct.sh
```

3. Parse performance data and saved in csv
```python
python parse_results_vllm.py --input_dir Qwen3-VL-235B-A22B-Instruct-logs --output_file Qwen3-VL-235B-A22B-Instruct-vllm_benchmark_results.csv
```
