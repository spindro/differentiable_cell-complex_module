import torch
from utils.cell_utils import compute_boundary, compute_Lup, compute_Lup_entmax, line_graph
from modules.dgm import DGM, LayerNorm
from modules.layers import GNN, MLP
from torch_geometric.utils import scatter

class DCM(torch.nn.Module):
    def __init__(self, hparams):
        super(DCM, self).__init__()
        self.hparams = hparams
        if hparams["use_gcn"]:
            self.graph_f = DGM(
                GNN(hparams["dgm_layers"], dropout=hparams["dropout"]),
                gamma=hparams["gamma"],
                std=hparams["std"],
            )
        else:
            self.graph_f = DGM(
                MLP(hparams["dgm_layers"], dropout=hparams["dropout"]),
                gamma=hparams["gamma"],
                std=hparams["std"],
            )
        self.poly_ln = LayerNorm(1)
    
    def forward(self, data):#x, edge_index=None, batch=None, edge_weight=None):
        x_aux = data["x"].detach()
        x_aux, edges, ne_probs = self.graph_f(x_aux, data["edge_index"], None)
        boundaries, row, col, xe, xe_aux, i, id_maps = compute_boundary(
            data["x"].detach(), x_aux.detach(), edges, max_k=self.hparams["k"]
        )
        Ldo, xe, xe_aux = line_graph(data["x"], row, col, i, xe, xe_aux)
        np_probs = None

        if self.hparams["sample_P"] == "fixed":
            Lup = compute_Lup(boundaries, id_maps).to(Ldo.device)
        else:  # self.hparams["sample_P"] == "entmax":
            Lup, poly_probs = compute_Lup_entmax(
                xe_aux, boundaries, id_maps, self.poly_ln, self.hparams["std"]
            )
            np_probs = scatter(
                poly_probs, row, dim=0, dim_size=data["x"].shape[0], reduce="sum"
            ) + scatter(poly_probs, col, dim=0, dim_size=data["x"].shape[0], reduce="sum")
        return {"x": data["x"], 
                "xe": xe, 
                "edges": edges, 
                "row": row, 
                "col": col, 
                "ne_probs": ne_probs, 
                "Ldo": Ldo, 
                "Lup": Lup, 
                "np_probs": np_probs}