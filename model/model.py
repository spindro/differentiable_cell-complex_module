import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from modules.layers import MLP
from modules.dcm import DCM
from modules.cell_network import CellNetwork
from torchmetrics import Accuracy


class ModelDCM(pl.LightningModule):
    def __init__(self, hparams):
        super(ModelDCM, self).__init__()
        self.save_hyperparameters(hparams)

        self.pre = MLP(
            hparams["pre_layers"], dropout=hparams["dropout"], final_activation=True
        )
        self.dcm = DCM(hparams)
        self.cell_network = CellNetwork(hparams)

        post_layers = hparams["post_layers"]
        post_layers[0] *= 2
        self.post = MLP(post_layers)
        self.avg_accuracy = None
        self.train_acc = Accuracy("multiclass", num_classes=post_layers[-1])
        self.val_acc = Accuracy("multiclass", num_classes=post_layers[-1])
        self.test_acc = Accuracy("multiclass", num_classes=post_layers[-1])

    def forward(self, x, edge_index=None, batch=None, edge_weight=None):
        x = self.pre(x)
        data = self.dcm({"x": x, "edge_index": edge_index})
        x = self.cell_network(data)
        return self.post(x), data["ne_probs"], data["np_probs"]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
        return optimizer

    def training_step(self, data, batch_idx):

        pred, edgelprobs, polylprobs = self(data.x, data.edge_index)
        pred = pred[data.train_mask]
        train_lab = data.y[data.train_mask]
        tr_loss = F.cross_entropy(pred, train_lab)

        corr_pred = (pred.argmax(-1) == train_lab.argmax(-1)).float().detach()
        if self.avg_accuracy is None:
            self.avg_accuracy = torch.ones_like(corr_pred) * 0.5

        tredgelprobs = edgelprobs[data.train_mask]
        point_w = self.avg_accuracy - corr_pred
        graph_loss = self.hparams["graph_loss_reg"] * (point_w * tredgelprobs).mean()
        tr_loss = tr_loss + graph_loss

        if polylprobs is not None:
            trpolylprobs = polylprobs[data.train_mask]
            poly_loss = self.hparams["poly_loss_reg"] * (point_w * trpolylprobs).mean()
            tr_loss = tr_loss + poly_loss

        self.avg_accuracy = (
            self.avg_accuracy.to(corr_pred.device) * 0.95 + 0.05 * corr_pred
        )

        self.train_acc(pred.softmax(-1), train_lab)

        self.log(
            "train_acc",
            self.train_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=data.x.shape[0],
        )
        self.log(
            "train_loss",
            tr_loss.detach(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=data.x.shape[0],
        )

        return tr_loss

    def validation_step(self, data, batch_idx):
        val_lab = data.y[data.val_mask]
        pred, _, _ = self(data.x, data.edge_index)

        for _ in range(1, self.hparams.ensemble_steps):
            pred_, _, _ = self(data.x, data.edge_index)
            pred += pred_

        self.val_acc(pred.softmax(-1)[data.val_mask], val_lab)
        self.log(
            "val_acc",
            self.val_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=val_lab.size(0),
        )

    def test_step(self, data, batch_idx):
        test_lab = data.y[data.test_mask]
        pred, _, _ = self(data.x, data.edge_index)

        for _ in range(1, self.hparams.ensemble_steps):
            pred_, _, _ = self(data.x, data.edge_index)
            pred += pred_

        self.test_acc(pred.softmax(-1)[data.test_mask], test_lab)
        self.log(
            "test_acc",
            self.test_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=test_lab.size(0),
        )
