## Qwen3-Omni-30B-A3B-Instruct PTPC FP8 Quantization with llm-compressor
1. Launch sglang docker image
```bash
docker run -it --ipc=host --network=host --privileged --security-opt seccomp=unconfined --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE --device=/dev/dri --device=/dev/mem -v /raid/users/xisun/:/root/workspace/ -v /raid/models:/models  --name lmsysorg_sglang_v0.5.6.post1-rocm700-mi30x lmsysorg/sglang:v0.5.6.post1-rocm700-mi30x
```

2. Install llm-compressor
```bash
pip install llmcompressor==0.9.0 transformers==4.57.1
```

3. Run Qwen3-Omni PTPC FP8 Quantization with llm-compressor
```python
python3 qwen3-omni_fp8_ptpc_quant.py
```
