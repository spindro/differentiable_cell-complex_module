import graph_tool as gt
import graph_tool.topology as top


gt.openmp_set_num_threads(18)
from itertools import combinations, compress

import networkx as nx
import torch
from entmax import entmax15
from torch import Tensor
from torch_geometric.utils import coalesce, remove_self_loops, scatter, to_undirected


# RING UTILS
def pyg_to_simplex_tree(edge_index: Tensor, size: int):
    st = []
    for v in range(size):
        st.append((v))
    for e in edge_index.cpu().T.numpy():
        st.append((e[0], e[1]))
    return st


def to_simplex_tree(size, row, col):
    st = []
    for v in range(size):
        st.append((v))
    for e in zip(row.cpu().numpy(), col.cpu().numpy()):
        st.append((e[0], e[1]))
    return st


def build_tables(simplex_tree, size):
    # Each of these data structures has a separate entry per dimension.
    id_maps = [{} for _ in range(2)]  # simplex -> id
    simplex_tables = [[] for _ in range(2)]  # matrix of simplices

    simplex_tables[0] = [[v] for v in range(size)]
    id_maps[0] = {tuple([v]): v for v in range(size)}

    for simplex in simplex_tree[size:]:
        # Assign this simplex the next unused ID
        next_id = len(simplex_tables[1])
        id_maps[1][tuple(simplex)] = next_id
        simplex_tables[1].append(simplex)
    return simplex_tables, id_maps


# ---- support for rings as cells Graph add_edge_list remove_parallel_edges


def get_rings(edge_index, max_k):
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()

    edge_list = edge_index.T
    graph_gt = gt.Graph(directed=False)
    graph_gt.add_edge_list(edge_list)
    gt.stats.remove_self_loops(graph_gt)
    gt.stats.remove_parallel_edges(graph_gt)

    rings = set()
    sorted_rings = set()
    for k in range(3, max_k + 1):
        pattern = nx.cycle_graph(k)
        pattern_edge_list = list(pattern.edges)
        pattern_gt = gt.Graph(directed=False)
        pattern_gt.add_edge_list(pattern_edge_list)
        sub_isos = top.subgraph_isomorphism(
            pattern_gt, graph_gt, induced=True, subgraph=True, generator=True
        )
        sub_iso_sets = map(lambda isomorphism: tuple(isomorphism.a), sub_isos)
        for iso in sub_iso_sets:
            if tuple(sorted(iso)) not in sorted_rings:
                rings.add(iso)
                sorted_rings.add(tuple(sorted(iso)))
    rings = list(rings)
    return rings


def build_tables_with_rings(edge_index, simplex_tree, size, max_k):
    # Build simplex tables and id_maps up to edges by conveniently
    # invoking the code for simplicial complexes
    cell_tables, id_maps = build_tables(simplex_tree, size)
    # Find rings in the graph
    rings = get_rings(edge_index, max_k=max_k)
    if len(rings) > 0:
        # Extend the tables with rings as 2-cells
        id_maps += [{}]
        cell_tables += [[]]
        assert len(cell_tables) == 3, cell_tables
        for cell in rings:
            next_id = len(cell_tables[2])
            id_maps[2][cell] = next_id
            cell_tables[2].append(list(cell))

    return cell_tables, id_maps


def get_ring_boundaries(ring):
    boundaries = list()
    for n in range(len(ring)):
        a = n
        if n + 1 == len(ring):
            b = 0
        else:
            b = n + 1
        boundaries.append(tuple(sorted([ring[a], ring[b]])))
    return sorted(boundaries)


def extract_boundaries_with_rings(simplex_tree, id_maps):

    boundaries = [{} for _ in range(3)]
    assert len(id_maps) <= 3
    if len(id_maps) == 3:
        boundaries += [{}]
        for cell in id_maps[2]:
            cell_boundaries = get_ring_boundaries(cell)
            boundaries[2][cell] = list()
            for boundary in cell_boundaries:
                assert boundary in id_maps[1], boundary
                boundaries[2][cell].append(boundary)

    return boundaries


def coalesced_ei(xo, xi, edge_index):
    N = xo.shape[0]
    _, xeo = coalesce(edge_index, xo[edge_index[0]] + xo[edge_index[1]], num_nodes=N)
    edge_index, xei = coalesce(
        edge_index, xi[edge_index[0]] + xi[edge_index[1]], num_nodes=N
    )
    row, col = edge_index
    # Compute node indices.
    mask = row < col
    row, col = row[mask], col[mask]
    i = torch.arange(row.size(0), dtype=torch.long, device=row.device)

    return row, col, xeo, xei, i


def line_graph(x, row, col, i, xo, xi):
    N = x.shape[0]
    (row, col), i = coalesce(
        torch.stack(
            [torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)], dim=0
        ),
        torch.cat([i, i], dim=0),
        N,
    )
    # Compute new edge indices according to `i`.
    count = scatter(torch.ones_like(row), row, dim=0, dim_size=N, reduce="sum")
    joints = torch.split(i, count.tolist())

    def generate_grid(x):
        row = x.view(-1, 1).repeat(1, x.numel()).view(-1)
        col = x.repeat(x.numel())
        return torch.stack([row, col], dim=0)

    joints = [generate_grid(joint) for joint in joints]
    joints = torch.cat(joints, dim=1)
    joints, _ = remove_self_loops(joints)
    NE = row.size(0) // 2
    if xo.shape[0] != i.shape[0]:
        s_ = i.shape[0]
        xeo = scatter(xo[:s_, :], i, dim=0, dim_size=NE, reduce="mean")
        xei = scatter(xi[:s_, :], i, dim=0, dim_size=NE, reduce="mean")
    else:
        xeo = scatter(xo, i, dim=0, dim_size=NE, reduce="mean")
        xei = scatter(xi, i, dim=0, dim_size=NE, reduce="mean")
    return coalesce(joints, num_nodes=NE), xeo, xei


def compute_boundary(xo, xi, edges, max_k):
    nV = xo.shape[0]
    row, col, xeo, xei, i = coalesced_ei(xo, xi, edges)
    simplex_tree = to_simplex_tree(nV, row, col)
    _, id_maps = build_tables_with_rings(edges, simplex_tree, nV, max_k)
    boundaries = extract_boundaries_with_rings(simplex_tree, id_maps)[2]
    return boundaries, row, col, xeo, xei, i, id_maps


def compute_Lup(boundaries, id_maps):
    comb = []
    edges = None
    if len(boundaries.values()) > 0:
        for val in boundaries.values():
            comb += combinations([id_maps[1][x] for x in val], 2)
        edges = to_undirected(torch.LongTensor(comb).t_())
    return edges


def compute_Lup_entmax(x, boundaries, id_maps, ln, std):
    comb = []
    vprobs = torch.zeros((len(boundaries),))
    poly_probs = torch.zeros((x.shape[0],))
    tmp_idx = []
    edges = None

    if len(boundaries.values()) > 0:
        for i, val in enumerate(boundaries.values()):
            idxs = [id_maps[1][x] for x in val]
            tmp_idx.append(idxs)
            vprobs[i] = torch.triu(torch.cdist(x[idxs], x[idxs]), diagonal=1).mean()
            poly_probs[idxs] += vprobs[i]

        vprobs = vprobs + torch.empty(vprobs.size()).normal_(mean=0, std=std)
        vprobs = entmax15(ln(vprobs))
        # vprobs = entmax_bisect(ln(vprobs), alpha=1.17)
        for valid in list(compress(tmp_idx, (vprobs > 0).tolist())):
            comb += combinations(valid, 2)

        edges = to_undirected(torch.LongTensor(comb).t_()).to(x.device)

    return edges, poly_probs
