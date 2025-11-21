## DeepSeek-R1-0528-MXFP4-ASQ
1. Launch sglang docker image
```bash
podman run -it --name rocm_sgl_dev_v0.5.5.post3-rocm700-mi35x-20251117  --device=/dev/dri --device=/dev/kfd --device=/dev/infiniband --device=/dev/infiniband/rdma_cm --privileged --network=host --ipc=host --cap-add=SYS_ADMIN --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --group-add keep-groups  -v /shared/amdgpu/home/share/models:/models -v /shared/amdgpu/home/share/training_models:/training_models -v /shared/data:/shared/data -v /shared:/shared  -v $HOME:/workdir --workdir /workdir  docker://rocm/sgl-dev:v0.5.5.post3-rocm700-mi35x-20251117 bash
```

2. Benchmark DeepSeek-R1-0528-MXFP4-ASQ with sglang
```bash
./test_DeepSeek-R1-0528-MXFP4-ASQ_sglang.sh
```

3. Parse performance data and saved in csv
```python
python parse_results_sglang.py --input_dir DeepSeek-R1-0528-MXFP4-ASQ-logs/ --output_file  DeepSeek-R1-0528-MXFP4-ASQ_benchmark_results.csv
```

## Qwen3-Omni-30B-A3B-Instruct (Text + Image) 
1. Benchmark Qwen3-Omni-30B-A3B-Instruct with sglang
```bash
./test_Qwen3-Omni-30B-A3B-Instruct_sglang.sh
```

2. Parse multimodal performance data and saved in csv
```python
python parse_results_sglang_multimodal.py --input_dir Qwen3-Omni-30B-A3B-Instruct-logs --output_file  Qwen3-Omni-30B-A3B-Instruct_benchmark_results.csv
```
