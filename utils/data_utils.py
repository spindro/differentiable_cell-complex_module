# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0


import os.path as osp
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, download_url, InMemoryDataset
from torch_sparse import coalesce


class WikipediaNetwork(InMemoryDataset):
    r"""

    The Wikipedia networks introduced in the
    `"Multi-scale Attributed Node Embedding"
    <https://arxiv.org/abs/1909.13021>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features represent several informative nouns in the Wikipedia pages.
    The task is to predict the average daily traffic of the web page.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"chameleon"`,
            :obj:`"crocodile"`, :obj:`"squirrel"`).
        geom_gcn_preprocess (bool): If set to :obj:`True`, will load the
            pre-processed data as introduced in the `"Geom-GCN: Geometric
            Graph Convolutional Networks" <https://arxiv.org/abs/2002.05287>_`,
            in which the average monthly traffic of the web page is converted
            into five categories to predict.
            If set to :obj:`True`, the dataset :obj:`"crocodile"` is not
            available.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    """

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = name.lower()
        assert self.name in ["chameleon", "squirrel"]
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> Union[str, List[str]]:
        return ["out1_node_feature_label.txt", "out1_graph_edges.txt"]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        pass

    def process(self):
        with open(self.raw_paths[0], "r") as f:
            data = f.read().split("\n")[1:-1]
        x = [[float(v) for v in r.split("\t")[1].split(",")] for r in data]
        x = torch.tensor(x, dtype=torch.float)
        y = [int(r.split("\t")[2]) for r in data]
        y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], "r") as f:
            data = f.read().split("\n")[1:-1]
            data = [[int(v) for v in r.split("\t")] for r in data]
        edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
        # Remove self-loops
        # edge_index, _ = remove_self_loops(edge_index)
        # Make the graph undirected
        # edge_index = to_undirected(edge_index)
        edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])


class WebKB(InMemoryDataset):
    r"""
    The WebKB datasets used in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features are the bag-of-words representation of web pages.
    The task is to classify the nodes into one of the five categories, student,
    project, course, staff, and faculty.
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cornell"`,
            :obj:`"Texas"` :obj:`"Washington"`, :obj:`"Wisconsin"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = (
        "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/"
        "1c4c04f93fa6ada91976cda8d7577eec0e3e5cce/new_data"
    )

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ["cornell", "texas", "washington", "wisconsin"]

        super(WebKB, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self):
        return ["out1_node_feature_label.txt", "out1_graph_edges.txt"]

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        for name in self.raw_file_names:
            download_url(f"{self.url}/{self.name}/{name}", self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], "r") as f:
            data = f.read().split("\n")[1:-1]
            x = [[float(v) for v in r.split("\t")[1].split(",")] for r in data]
            x = torch.tensor(x, dtype=torch.float32)

            y = [int(r.split("\t")[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], "r") as f:
            data = f.read().split("\n")[1:-1]
            data = [[int(v) for v in r.split("\t")] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            # edge_index = to_undirected(edge_index)
            # We also remove self-loops in these datasets in order not to mess up.
            # edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return "{}()".format(self.name)


def get_hetero_dataset(name):
    if name in ["texas", "wisconsin"]:
        dataset = WebKB(root="data/Hetero", name=name, transform=T.NormalizeFeatures())
    elif name in ["chameleon", "squirrel"]:
        dataset = WikipediaNetwork(
            root="data/Hetero", name=name, transform=T.NormalizeFeatures()
        )
    else:
        raise ValueError(f"dataset {name} not supported in dataloader")

    return dataset


def cross_validation_split(data, dataset_name=None, curr_seed=0):

    loaded_data = np.load(f"./data_splits/{dataset_name}/splits.npz", allow_pickle=True)
    final_splits = loaded_data["splits"].item()

    n_nodes = data.y.shape[0]
    # Get current split
    train_indices = torch.as_tensor(final_splits[curr_seed]["Train_idx"])
    val_indices = torch.as_tensor(final_splits[curr_seed]["Test_idx"])
    test_indices = torch.as_tensor(final_splits[curr_seed]["Test_idx"])

    device = data.y.device
    train_mask = torch.zeros(n_nodes, dtype=torch.bool).to(device)
    train_mask[train_indices] = True

    valid_mask = torch.zeros(n_nodes, dtype=torch.bool).to(device)
    valid_mask[val_indices] = True

    test_mask = torch.zeros(n_nodes, dtype=torch.bool).to(device)
    test_mask[test_indices] = True

    data.train_mask = train_mask
    data.val_mask = valid_mask
    data.test_mask = test_mask

    return data
