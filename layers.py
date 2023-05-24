import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MLP(nn.Module):
    def __init__(self, layers_size, dropout=0.0, final_activation=False):
        super(MLP, self).__init__()
        layers = []
        for li in range(1, len(layers_size)):
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(layers_size[li - 1], layers_size[li]))
            if li == len(layers_size) - 1 and not final_activation:
                continue
            layers.append(nn.ReLU())
        self.MLP = nn.Sequential(*layers)

    def forward(self, x, e=None):
        x = self.MLP(x)
        return x


class GNN(nn.Module):
    def __init__(self, layers_size, dropout=0.0, last_act=False):
        super(GNN, self).__init__()
        self.d = dropout
        self.convs = nn.ModuleList()
        self.last_act = last_act
        for li in range(1, len(layers_size)):
            self.convs.append(GCNConv(layers_size[li - 1], layers_size[li]))

    def forward(self, x, e):
        for i, c in enumerate(self.convs):
            x = c(F.dropout(x, p=self.d, training=self.training), e)
            if i == len(self.convs) and self.last_act is False:
                break
            x = x.relu()
        return x


class CW(nn.Module):
    def __init__(self, F_in, F_out):
        super(CW, self).__init__()
        self.har = nn.Linear(F_in, F_out)
        self.sol = GCNConv(F_in, F_out, add_self_loops=False)
        self.irr = GCNConv(F_in, F_out, add_self_loops=False)

    def forward(self, xe, Lu, Ld):
        z_h = self.har(xe)
        z_s = self.sol(xe, Lu)
        z_i = self.irr(xe, Ld)
        return z_h + z_s + z_i


class CWNN(nn.Module):
    def __init__(self, layers_size, dropout=0.0, last_act=False):
        super(CWNN, self).__init__()
        self.d = dropout
        self.convs = nn.ModuleList()
        self.last_act = last_act
        for li in range(1, len(layers_size)):
            self.convs.append(CW(layers_size[li - 1], layers_size[li]))

    def forward(self, x, Lu, Ld):
        for i, c in enumerate(self.convs):
            x = c(F.dropout(x, p=self.d, training=self.training), Lu, Ld)
            if i == len(self.convs) and self.last_act is False:
                break
            x = x.relu()
        return x
