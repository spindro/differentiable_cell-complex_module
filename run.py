import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pytorch_lightning as pl
import torch
import wandb
from data_utils import get_hetero_dataset
from model import DCM
from pytorch_lightning.loggers import WandbLogger
from torch_geometric import seed_everything
from torch_geometric.data import LightningNodeData
from torch_geometric.datasets import Coauthor, Planetoid
from torch_geometric.transforms import NormalizeFeatures, RandomNodeSplit


torch.set_default_tensor_type("torch.cuda.FloatTensor")
torch.set_float32_matmul_precision("high")

DATA_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
sweep_config = {
    "method": "grid",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "dataset": {
            "values": [
                "cora",
                "citeseer",
                # "pubmed",
                # "cs",
                # "physics",
                "texas",
                "wisconsin",
                # "squirrel",
                # "chameleon",
            ]
        },
        "seed": {"values": [42]},
        "data_seed": {"values": DATA_SEEDS},
        "hsize": {"values": [32]},
        "n_pre": {"values": [1]},
        "n_post": {"values": [1]},
        "n_conv": {"values": [1]},
        "n_dgm_layers": {"values": [2]},
        "dropout": {"values": [0.5]},
        "lr": {"values": [0.01]},
        "use_gcn": {"values": [False]},
        "k": {"values": [4]},
        "graph_loss_reg": {"values": [1]},
        "poly_loss_reg": {"values": [1]},
    },
}


def sweep_iteration():

    wandb.init()
    wlog = WandbLogger()
    seed_everything(wandb.config.seed)

    if wandb.config.dataset in ["physics", "cs"]:
        dataset = Coauthor(
            root="data/Coauthor", name=wandb.config.dataset, transform=RandomNodeSplit()
        )
    elif wandb.config.dataset in ["cora", "citeseer", "pubmed"]:
        dataset = Planetoid(
            root="data/Planetoid",
            name=wandb.config.dataset,
            split="full",
            transform=NormalizeFeatures(),
        )
    elif wandb.config.dataset in ["squirrel", "texas", "chameleon", "wisconsin"]:
        dataset = get_hetero_dataset(wandb.config.dataset)
    else:
        raise ValueError("Dataset not found")

    data = dataset[0]

    datamodule = LightningNodeData(
        data,
        data.train_mask,
        data.val_mask,
        data.test_mask,
        loader="full",
        generator=torch.Generator(device="cuda"),
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_acc", mode="max")

    std = 0
    ensemble_steps = 1
    if wandb.config.dataset in ["physics", "cs"]:
        torch.set_float32_matmul_precision("medium")
        gamma = 70
        epochs = 100
    elif wandb.config.dataset in ["pubmed", "cora"]:
        gamma = 50
        epochs = 200
    elif wandb.config.dataset in ["citeseer"]:
        gamma = 20
        epochs = 200
    elif wandb.config.dataset in ["texas"]:
        gamma = 5
        epochs = 200
    elif wandb.config.dataset in ["wisconsin"]:
        gamma = 10
        epochs = 200
    elif wandb.config.dataset in ["squirrel", "chameleon"]:
        std = 0.1
        ensemble_steps = 5
        gamma = 20
        epochs = 200
    else:
        raise ValueError("Dataset not found")

    trainer = pl.Trainer(
        logger=wlog,
        accelerator="gpu",
        devices=1,
        max_epochs=epochs,
        log_every_n_steps=3,
        check_val_every_n_epoch=3,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback],
    )

    hsize = wandb.config.hsize
    hyperparams = {
        "num_features": dataset.num_features,
        "num_classes": dataset.num_classes,
        "pre_layers": [dataset.num_features]
        + [hsize for _ in range(wandb.config.n_pre)],
        "post_layers": [hsize for _ in range(wandb.config.n_post)]
        + [dataset.num_classes],
        "dgm_layers": [hsize for _ in range(wandb.config.n_dgm_layers + 1)],
        "conv_layers": [hsize for _ in range(wandb.config.n_conv)],
        "lr": wandb.config.lr,
        "use_gcn": wandb.config.use_gcn,
        "dropout": wandb.config.dropout,
        "k": wandb.config.k,
        "gamma": gamma,
        "std": std,
        "graph_loss_reg": wandb.config.graph_loss_reg,
        "poly_loss_reg": wandb.config.poly_loss_reg,
        "ensemble_steps": ensemble_steps,
    }

    model = DCM(hyperparams)
    trainer.fit(model, datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="DCM")
    wandb.agent(sweep_id, function=sweep_iteration)
    wandb.finish()
