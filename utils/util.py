import numpy as np
import networkx as nx
from collections import defaultdict


def load_data(file_name, p):
    data = np.load('%s_%.1f.npz' % (file_name, p))
    edge_index1, edge_index2 = data['edge_index1'].astype(np.int64), data['edge_index2'].astype(np.int64)
    anchor_links, test_pairs = data['pos_pairs'].astype(np.int64), data['test_pairs'].astype(np.int64)
    x1, x2 = data['x1'].astype(np.float32), data['x2'].astype(np.float32)

    return edge_index1, edge_index2, x1, x2, anchor_links.T, test_pairs.T


def get_neighbors(edge_index, anchor_nodes):
    g = nx.Graph()
    g.add_edges_from(edge_index.T)

    context_pairs = []

    for node in anchor_nodes:
        neighbors = [n for n in g.neighbors(node)]
        if len(neighbors) > 100:
            degree = np.array([x[1] for x in list(nx.degree(g, neighbors))])
            p = degree / np.sum(degree)
            neighbors = np.random.choice(neighbors, 100, p=p)
        for n in neighbors:
            context_pairs.append([node, n])
    context_pairs = np.array(context_pairs, dtype=np.int64)
    np.random.shuffle(context_pairs)
    return context_pairs


def neg_context_prob(edge_index, anchor_nodes, num_nodes):
    neighbor_dict = defaultdict(list)
    row, col = edge_index
    for i in range(len(row)):
        if row[i] in anchor_nodes:
            neighbor_dict[row[i]].append(col[i])

    prob = np.ones((len(anchor_nodes), num_nodes))
    for i, anchor in enumerate(anchor_nodes):
        idx = neighbor_dict[anchor]
        prob[i][idx] = 0

    anchor_node_map = -1 * np.ones(num_nodes, dtype=np.int64)
    anchor_node_map[anchor_nodes] = np.arange(len(anchor_nodes), dtype=np.int64)

    return prob, anchor_node_map


def balance_inputs(context_pairs1, context_pairs2):
    if len(context_pairs1) < len(context_pairs2):
        len_diff = len(context_pairs2) - len(context_pairs1)
        idx = np.random.choice(len(context_pairs1), len_diff)
        imputes = context_pairs1[idx]
        context_pairs1 = np.vstack([context_pairs1, imputes])
    else:
        len_diff = len(context_pairs1) - len(context_pairs2)
        idx = np.random.choice(len(context_pairs2), len_diff)
        imputes = context_pairs2[idx]
        context_pairs2 = np.vstack([context_pairs2, imputes])

    return context_pairs1, context_pairs2


def anchor_as_attr(x1, x2, anchor_links):
    n1, d1 = x1.shape
    n2, d2 = x2.shape
    n_anchor = len(anchor_links[0])
    x1_t = np.zeros((n1, n_anchor), dtype=np.float32)
    x2_t = np.zeros((n2, n_anchor), dtype=np.float32)
    row, col = anchor_links
    for i in range(len(row)):
        x1_t[row[i]][i] = 1
        x2_t[col[i]][i] = 1
    if n1 == d1 and (np.diag(x1) == 1).all():
        # graph has no node attributes but an identity matrix
        x1 = x1_t
    else:
        x1 = np.concatenate([x1, x1_t], axis=1)
    if n2 == d2 and (np.diag(x2) == 1).all():
        x2 = x2_t
    else:
        x2 = np.concatenate([x2, x2_t], axis=1)

    return x1, x2
