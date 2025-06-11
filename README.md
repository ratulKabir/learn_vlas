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