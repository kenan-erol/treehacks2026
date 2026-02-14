uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 uv pip install --editable .