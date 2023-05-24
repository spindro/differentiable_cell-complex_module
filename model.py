import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from cell_utils import compute_boundary, compute_Lup, compute_Lup_entmax, line_graph
from dgm import DGM, LayerNorm
from layers import CWNN, GNN, MLP
from torch_geometric.utils import scatter
from torchmetrics import Accuracy


class DCM(pl.LightningModule):
    def __init__(self, hparams):
        super(DCM, self).__init__()
        self.save_hyperparameters(hparams)

        self.pre = MLP(
            hparams["pre_layers"], dropout=hparams["dropout"], final_activation=True
        )

        if hparams["use_gcn"]:
            self.graph_f = DGM(
                GNN(hparams["dgm_layers"], dropout=hparams["dropout"]),
                sampler=hparams["sampler"],
                gamma=hparams["gamma"],
                std=hparams["std"],
            )
        else:
            self.graph_f = DGM(
                MLP(hparams["dgm_layers"], dropout=hparams["dropout"]),
                sampler=hparams["sampler"],
                gamma=hparams["gamma"],
                std=hparams["std"],
            )

        self.gnn = GNN(hparams["conv_layers"], dropout=hparams["dropout"])
        self.cwnn = CWNN(hparams["conv_layers"])

        self.poly_ln = LayerNorm(1)
        post_layers = hparams["post_layers"]
        post_layers[0] *= 2
        self.post = MLP(post_layers)
        self.avg_accuracy = None
        self.train_acc = Accuracy("multiclass", num_classes=post_layers[-1])
        self.val_acc = Accuracy("multiclass", num_classes=post_layers[-1])
        self.test_acc = Accuracy("multiclass", num_classes=post_layers[-1])

    def forward(self, x, edge_index=None, batch=None, edge_weight=None):
        x = self.pre(x)
        x_aux = x.detach()
        x_aux, edges, ne_probs = self.graph_f(x_aux, edge_index, None)
        boundaries, row, col, xe, xe_aux, i, id_maps = compute_boundary(
            x.detach(), x_aux.detach(), edges, max_k=self.hparams["k"]
        )
        Ldo, xe, xe_aux = line_graph(x, row, col, i, xe, xe_aux)
        np_probs = None

        if self.hparams["sample_P"] == "fixed":
            Lup = compute_Lup(boundaries, id_maps).to(Ldo.device)
        else:  # self.hparams["sample_P"] == "entmax":
            Lup, poly_probs = compute_Lup_entmax(
                xe_aux, boundaries, id_maps, self.poly_ln, self.hparams["std"]
            )
            np_probs = scatter(
                poly_probs, row, dim=0, dim_size=x.shape[0], reduce="sum"
            ) + scatter(poly_probs, col, dim=0, dim_size=x.shape[0], reduce="sum")

        x = self.gnn(x, edges).relu()
        xe = self.cwnn(xe, Ldo, Lup).relu()
        xed = scatter(xe, row, dim=0, dim_size=x.shape[0], reduce="sum") + scatter(
            xe, col, dim=0, dim_size=x.shape[0], reduce="sum"
        )
        x = torch.cat([x, xed], dim=-1)

        return self.post(x), ne_probs, np_probs

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
