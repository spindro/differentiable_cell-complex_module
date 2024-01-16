import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pytorch_lightning as pl
import torch
import wandb
from model.model_dcm import ModelDCM
from pytorch_lightning.loggers import WandbLogger
from torch_geometric import seed_everything
from torch_geometric.data import LightningNodeData
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


torch.set_default_tensor_type("torch.cuda.FloatTensor")
torch.set_float32_matmul_precision("high")

config = {
    "metric": {"name": "val_acc", "goal": "maximize"},
    "seed":  42,
    "data_seed":  0,
    "hsize":  32,
    "n_pre":  1,
    "n_post":  1,
    "n_conv":  1,
    "n_dgm_layers":  2,
    "dropout":  0.5,
    "lr":  0.01,
    "use_gcn":  False,
    "k":  4,
    "graph_loss_reg":  1,
    "poly_loss_reg":  1,
    }


def train(config):
    wandb.init()
    wlog = WandbLogger()
    seed_everything(config["seed"])

    dataset = Planetoid(
        root="data/Planetoid",
        name="cora",
        split="full",
        transform=NormalizeFeatures(),
    )
    
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

    gamma = 50
    epochs = 200

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

    hsize = config["hsize"]
    hyperparams = {
        "num_features": dataset.num_features,
        "num_classes": dataset.num_classes,
        "pre_layers": [dataset.num_features]
        + [hsize for _ in range(config["n_pre"])],
        "post_layers": [hsize for _ in range(config["n_post"])]
        + [dataset.num_classes],
        "dgm_layers": [hsize for _ in range(config["n_dgm_layers"] + 1)],
        "conv_layers": [hsize for _ in range(config["n_conv"])],
        "lr": config["lr"],
        "use_gcn": config["use_gcn"],
        "dropout": config["dropout"],
        "k": config["k"],
        "gamma": gamma,
        "std": std,
        "graph_loss_reg": config["graph_loss_reg"],
        "poly_loss_reg": config["poly_loss_reg"],
        "ensemble_steps": ensemble_steps,
    }

    model = ModelDCM(hyperparams)
    trainer.fit(model, datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)

if __name__ == "__main__":
    train(config)
    wandb.finish()
