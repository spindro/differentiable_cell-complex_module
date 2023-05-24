import torch
import torch.nn as nn
from entmax import entmax15  # , entmax_bisect
from torch_geometric.utils import to_undirected


# TODO check from my_entmax import entmax15


class LayerNorm(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = nn.Parameter(gamma * torch.ones(1))  # std 10*
        self.beta = nn.Parameter(torch.zeros(1))  # mean
        self.eps = 1e-6

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        if x.size(-1) == 1:
            std = 1
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def top_k(x: torch.tensor, k: int = 3, temperature=1, tau=1):
    """
    Gumbel top K sampling
    """
    n, _ = x.shape
    # get probs and logits
    probs = torch.exp(-temperature * torch.cdist(x, x))
    logits = torch.log(probs + 1e-9)

    # Sample
    q = torch.rand_like(probs)
    lq = (logits - torch.log(-torch.log(q + 1e-9))) / tau

    if lq.shape[0] > 30000:
        lq = lq.fill_diagonal_(-1e-9)
        logprobs1, indices1 = torch.topk(lq[:15000, :15000], k)
        logprobs2, indices2 = torch.topk(lq[15000:, 15000:], k)
        logprobs, indices = torch.cat((logprobs1, logprobs2), dim=0), torch.cat(
            (indices1, indices2), dim=0
        )
    else:
        logprobs, indices = torch.topk(lq.fill_diagonal_(-1e-9), k)

    # Edges
    rows = torch.arange(n).view(n, 1).to(logits.device).repeat(1, k)
    edges = torch.stack((indices.view(-1), rows.view(-1)), -2)
    return to_undirected(edges), logprobs.sum(dim=1)


def entmax(x: torch.tensor, ln, std=0):
    probs = -torch.cdist(x, x)
    probs = probs + torch.empty(probs.size()).normal_(mean=0, std=std)
    vprobs = ln(probs)

    if vprobs.shape[0] > 30000:
        vprobs = vprobs.fill_diagonal_(-1e-9)
        vprobs1 = entmax15(
            vprobs[: vprobs.shape[0] // 2, : vprobs.shape[0] // 2], dim=-1
        )
        vprobs2 = entmax15(
            vprobs[: vprobs.shape[0] // 2, vprobs.shape[0] // 2 :], dim=-1
        )
        vprobs3 = entmax15(
            vprobs[vprobs.shape[0] // 2 :, : vprobs.shape[0] // 2], dim=-1
        )
        vprobs4 = entmax15(
            vprobs[vprobs.shape[0] // 2 :, vprobs.shape[0] // 2 :], dim=-1
        )
        vprobs_v1 = torch.vstack((vprobs1, vprobs3))
        vprobs_v2 = torch.vstack((vprobs2, vprobs4))
        vprobs = torch.hstack((vprobs_v1, vprobs_v2))
    else:
        vprobs = entmax15(vprobs.fill_diagonal_(-1e-6), dim=-1)

    res = (((vprobs + vprobs.t()) / 2) > 0) * 1
    edges = res.nonzero().t_()
    logprobs = res.sum(dim=1)
    return edges, logprobs


class DGM(nn.Module):
    def __init__(self, embed_f: nn.Module, sampler="entmax", gamma=10, std=0):
        super(DGM, self).__init__()
        self.ln = LayerNorm(gamma)
        self.std = std
        self.embed_f = embed_f
        self.sampler = sampler

    def forward(self, x, edges, placeholder=None):
        # Input embedding
        x = self.embed_f(x, edges)
        if self.sampler == "entmax":
            edges_hat, logprobs = entmax(x=x, ln=self.ln, std=self.std)

        if self.sampler == "top_k":
            edges_hat, logprobs = top_k(x=x)

        return x, edges_hat, logprobs
