docker run --gpus all \
  -p 8000:8000 \
  --ipc=host \
  -v ~/models:/models \
  vllm/vllm-openai:latest \
  --model /models/Cosmos-Reason2-2B \
  --trust-remote-code \
  --max-model-len 4096 \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.6