# DCM

## Installation and Execution
```
conda create --name dcm --file conda_env.txt
conda activate dcm
pip install pytorch-lightning==1.9.5 wandb==0.13.9 entmax==1.1 
python train.py
```
Please cite:

```
@inproceedings{
battiloro2024from,
title={From Latent Graph to Latent Topology Inference: Differentiable Cell Complex Module},
author={Claudio Battiloro and Indro Spinelli and Lev Telyatnikov and Michael M. Bronstein and Simone Scardapane and Paolo Di Lorenzo},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=0JsRZEGZ7L}
}

```
