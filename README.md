# learn_vlas
To learn more about the vision language actions models
## Setup Instructions

To set up the environment and run `scripts/octo_inference.py`, follow these steps:

```bash
git clone https://github.com/octo-models/octo.git
cd octo
conda create -n octo python=3.10
conda activate octo
pip install -e .
pip install -r requirements.txt
pip install --upgrade "jax[cuda12_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install scipy==1.11.0
```

> **Note:** Adjust the JAX installation command according to your CUDA version.

## Running SmolVLM Models

To set up the environment for SmolVLM models:

```bash
conda create -n smolvlm python=3.10
conda activate smolvlm
pip install -r requirements.txt
# Install PyTorch for your CUDA version, e.g., for CUDA 12.4:
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn --no-build-isolation
```

> **Note:** Replace the PyTorch install command with the one matching your CUDA version from [PyTorch Get Started](https://pytorch.org/get-started/locally/).