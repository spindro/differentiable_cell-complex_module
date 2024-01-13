import torch
import torch.nn as nn
from entmax import entmax15


class LayerNorm(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = nn.Parameter(gamma * torch.ones(1)) 
        self.beta = nn.Parameter(torch.zeros(1))
        self.eps = 1e-6

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        if x.size(-1) == 1:
            std = 1
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def entmax(x: torch.tensor, ln, std=0):
    probs = -torch.cdist(x, x)
    probs = probs + torch.empty(probs.size()).normal_(mean=0, std=std)
    vprobs = entmax15(ln(probs).fill_diagonal_(-1e-6), dim=-1)
    res = (((vprobs + vprobs.t()) / 2) > 0) * 1
    edges = res.nonzero().t_()
    logprobs = res.sum(dim=1)
    return edges, logprobs


class DGM(nn.Module):
    def __init__(self, embed_f: nn.Module, gamma=10, std=0):
        super(DGM, self).__init__()
        self.ln = LayerNorm(gamma)
        self.std = std
        self.embed_f = embed_f

    def forward(self, x, edges, placeholder=None):
        # Input embedding
        x = self.embed_f(x, edges)
        edges_hat, logprobs = entmax(x=x, ln=self.ln, std=self.std)

        return x, edges_hat, logprobs
