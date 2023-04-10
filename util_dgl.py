import argparse
from datetime import datetime
import logging
import os
import sys
import time

import dgl
from numba import jit
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import get_free_gpu
from layers import TSAGEConv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", dest="gpu", action="store_true",
                        help="Whether use GPU.")
    parser.add_argument("--no-gpu", dest="gpu", action="store_false",
                        help="Whether use GPU.")
    return parser.parse_args(["--gpu"])


def set_logger(log_file=False):
    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(
            'log/dgl-{}.log'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print("%r  %2.2f s" % (method.__name__, te - ts))
        return result
    return timed

@timeit
# @jit
def compute_degrees(new_node_ids: list, num):
    degs = np.zeros(num)
    for node_ids in new_node_ids:
        for i, idx in enumerate(node_ids):
            degs[idx] = i + 1
    return degs

@timeit
# @jit
def construct_adj(src, dst, t, num):
    adj_eid_l = [[] for _ in range(num)]
    adj_ngh_l = [[] for _ in range(num)]
    adj_ts_l = [[] for _ in range(num)]
    for i in range(len(src)):
        adj_eid_l[src[i]].append(i)
        adj_ngh_l[src[i]].append(dst[i])
        adj_ts_l[src[i]].append(t[i])

        if src[i] == dst[i]:
            continue
        adj_eid_l[dst[i]].append(i)
        adj_ngh_l[dst[i]].append(src[i])
        adj_ts_l[dst[i]].append(t[i])
    
    adj_eid_l = [np.array(e) for e in adj_eid_l]
    adj_ngh_l = [np.array(e) for e in adj_ngh_l]
    adj_ts_l = [np.array(e) for e in adj_ts_l]
    return adj_eid_l, adj_ngh_l, adj_ts_l

def construct_dglgraph(edges, nodes, device, node_dim=128, bidirected=True):
    ''' Edges should be a pandas DataFrame, and its columns should be columns
    comprise of  from_node_id, to_node_id, timestamp, state_label, features_separated_by_comma.
    Here `state_label` varies in edge classification tasks.

    Nodes should be a pandas DataFrame, and its columns should be columns comprise
    of node_id, id_map, role, label, features_separated_by_comma.

    By default, we use the single directional edges to store the bi-directional
    edge messages for memory reduction. If `bidirected` is set `True`, we add
    the inverse edges into the DGLGraph. In this case, we retain edges in the
    increasing temporal order.
    '''
    src = edges["from_node_id"]
    dst = edges["to_node_id"]
    etime = torch.tensor(edges["timestamp"], device=device)
    efeature = torch.tensor(edges.iloc[:, 4:].to_numpy(), device=device) if len(
        edges.columns) > 4 else torch.zeros((len(edges), 1), device=device)

    if len(nodes.columns) > 4:
        nfeature = torch.tensor(nodes.iloc[:, 4:].to_numpy(), device=device)
    else:
        nfeature = nn.Parameter(nn.init.xavier_normal_(
            torch.empty(len(nodes), node_dim, device=device)))

    if bidirected:
        # In this way, we repeat the edge one by one, remaining the increasing
        # temporal order.
        u = np.vstack((src, dst)).transpose().flatten()
        v = np.vstack((dst, src)).transpose().flatten()
        src, dst = u, v
        etime = etime.repeat_interleave(2)
        efeature = efeature.repeat_interleave(2, dim=0)
    # Adding edges in the time increasing order, so that `group_apply_edges`
    # will process the neighbors temporally ascendingly. Further we store both
    # source and destionation node representations at timestamp t on the same
    # edge `(u, v, t)`.
    # For bi-partite graphs, we can only access the node temporal features from
    # one of source and destination tensors `g.edata["src_feat{layer}"][eid]`.
    # For bidirected graphs, we use two edges `(u, v, t)` and `(v, u, t)`. Let
    # `eid1` denote the eid of `(u, v, t)`, and `eid2` denote the eid of
    # `(v, u, t)`. We can access node temporal features `(u, t)` from both
    # tensors by `g.edata["src_feat{layer}"][eid1]` and
    # `g.edata["dst_feat{layer}"][eid2]`.
    # g = dgl.graph((src, dst)).to(device)
    g = dgl.DGLGraph((src, dst)).to(device)
    g.ndata["nfeat"] = nfeature  # .to(device)
    g.edata["timestamp"] = etime.to(nfeature)  # .to(device)
    g.edata["efeat"] = efeature.to(nfeature)  # .to(device)
    return g


def prepare_mp(g):
    """
    Explicitly materialize the CSR, CSC and COO representation of the given graph
    so that they could be shared via copy-on-write to sampler workers and GPU
    trainers.

    This is a workaround before full shared memory support on heterogeneous graphs.
    """
    g.in_degree(0)
    g.out_degree(0)
    g.find_edges([0])


def test_graph():
    """Load data using ``minibatch.load_data()``. If name is not given, we return a sample graph with 10 nodes and 45 edges, which is a complete graph.
    """
    nodes = pd.DataFrame(columns=["node_id", "id_map", "role", "label"])
    nodes["node_id"] = np.arange(10)
    nodes["id_map"] = np.arange(10)
    nodes["role"] = 0
    nodes["label"] = 0
    edges = pd.DataFrame(
        columns=["from_node_id", "to_node_id", "timestamp", "state_label"])
    edges["from_node_id"] = np.concatenate([
        [i for _ in range(9 - i)] for i in range(10)])
    edges["to_node_id"] = np.concatenate([
        [j for j in range(i + 1, 10)] for i in range(10)])
    edges["timestamp"] = np.arange(45, dtype=np.float)
    edges["state_label"] = 0
    dtypes = edges.dtypes
    dtypes[["from_node_id", "to_node_id"]] = int
    edges = edges.astype(dtypes)
    return edges, nodes


def padding_node(edges, nodes):
    if 0 in set(nodes["id_map"]):
        return edges, nodes
    print("padding node 0")
    nodes.loc[len(nodes)] = [0] * len(nodes.columns)
    dtypes = nodes.dtypes
    dtypes[["id_map"]] = int
    nodes = nodes.astype(dtypes).sort_values(
        by="id_map").reset_index(drop=True)

    # delta = edges["timestamp"].shift(-1) - edges["timestamp"]
    # assert np.all(delta.loc[:len(delta)-1] >= 0)
    edges["timestamp"] = edges["timestamp"] - \
        edges["timestamp"].min() + 1e-6  # assume time positive
    edges.loc[len(edges)] = [0] * len(edges.columns)
    dtypes = edges.dtypes
    dtypes[["from_node_id", "to_node_id"]] = int
    edges = edges.astype(dtypes).sort_values(
        by="timestamp").reset_index(drop=True)
    # delta = edges["timestamp"].shift(-1) - edges["timestamp"]
    # assert np.all(delta[:len(delta)-1] >= 0)
    return edges, nodes


def main():
    args = parse_args()
    logger = set_logger()
    logger.info(args)
    if args.gpu:
        device = torch.device("cuda:{}".format(get_free_gpu()))
    else:
        device = torch.device("cpu")

    edges, nodes = test_graph()
    g = construct_dglgraph(edges, nodes, device)

    logger.info(f"Begin Conv. Device {device}")
    dim = g.ndata["nfeat"].shape[-1]
    dims = [dim, 108, 4]
    # for l in range(1, 3):
    #     logger.info(f"Graph Conv Layer {l}.")
    #     model = TSAGEConv(in_feats=dims[l-1], out_feats=dims[l], aggregator_type="mean")
    #     model = model.to(device)
    #     src_feat, dst_feat = model(g, current_layer=l)
    #     g.edata[f"src_feat{l}"] = src_feat
    #     g.edata[f"dst_feat{l}"] = dst_feat
    model = TSAGEConv(in_feats=dims[0],
                      out_feats=dims[1], aggregator_type="mean")
    model = model.to(device)
    import copy
    nfeat_copy = copy.deepcopy(g.ndata["nfeat"])
    loss_fn = nn.CosineEmbeddingLoss(margin=0.5)
    import itertools
    optimizer = torch.optim.Adam(itertools.chain(
        [g.ndata["nfeat"]], model.parameters()), lr=0.01)
    # print(nfeat_copy)
    for i in range(10):
        logger.info("Epoch %3d", i)
        model.train()
        optimizer.zero_grad()
        src_feat, dst_feat = model(g, current_layer=1)
        labels = torch.ones((g.number_of_edges()), device=device)
        loss = loss_fn(src_feat, dst_feat, labels)
        loss.backward()
        optimizer.step()
        print("nfeat")
        print(g.ndata["nfeat"].storage().data_ptr())
        print("nfeat copy")
        print(nfeat_copy.storage().data_ptr())
        assert not torch.all(torch.eq(nfeat_copy, g.ndata["nfeat"]))
    print(src_feat.shape, dst_feat.shape)
    # z = src_feat.sum()
    # z.backward()
    print(g.ndata["nfeat"].grad)
    return src_feat, dst_feat


def nodeflow_test():
    pass


def fullgraph_test():
    pass


def subgraph_test():
    pass


if __name__ == "__main__":
    main()
