import torch
from modules.layers import CWNN, GNN
from torch_geometric.utils import scatter

class CellNetwork(torch.nn.Module):
    def __init__(self, hparams):
        super(CellNetwork, self).__init__()
        self.gnn = GNN(hparams["conv_layers"], dropout=hparams["dropout"])
        self.cwnn = CWNN(hparams["conv_layers"])
        
    def forward(self, data):
        x = self.gnn(data["x"], data["edges"]).relu()
        xe = self.cwnn(data["xe"], data["Ldo"], data["Lup"]).relu()
        xed = scatter(xe, data["row"], dim=0, dim_size=x.shape[0], reduce="sum") + scatter(
            xe, data["col"], dim=0, dim_size=x.shape[0], reduce="sum"
        )
        x = torch.cat([x, xed], dim=-1)
        return x