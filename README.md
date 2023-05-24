# DCM

## Installation
```
conda create --name dcm --yes
conda activate dcm
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia --yes
pip install pytorch-lightning torchfunc wandb entmax torch_sparse torch_scatter
conda install pyg -c pyg --yes
conda install -c conda-forge graph-tool --yes
conda install ipykernel ipywidgets networkx jupyter
```
