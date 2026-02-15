uv venv --python 3.12 --seed
source .venv/bin/activate
PIP_CONSTRAINT="" pip install "vllm==0.8.5.post1" --no-deps
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='nvidia/Cosmos-Reason2-2B', local_dir='/home/asus/models/Cosmos-Reason2-2B')
"
# uv pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-nccl-cu12
# uv pip install --index-url https://pypi.nvidia.com torch torchvision torchaudio
# uv pip install torch torchvision torchaudio --index-url https://pypi.nvidia.com
# # Verify CUDA is available
# python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
# uv pip install vllm --no-deps
# uv pip install -r <(pip show vllm 2>/dev/null | grep Requires | sed 's/Requires: //' | tr ',' '\n' | grep -v torch)
