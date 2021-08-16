import numpy as np


def negative_sampling_uniform(anchor1, anchor2, n1, n2, n_negs):
    anchor1 = anchor1.cpu().detach().numpy()
    anchor2 = anchor2.cpu().detach().numpy()
    num_anchors = len(anchor1)
    p1 = np.ones((num_anchors, n2))
    p2 = np.ones((num_anchors, n1))
    negs1, negs2 = [], []
    for i in range(num_anchors):
        p = p1[i]
        p[anchor2[i]] = 0
        p = p / np.sum(p)
        samples = np.random.choice(n2, n_negs, p=p)
        negs1.append(list(samples))

        p = p2[i]
        p[anchor1[i]] = 0
        p = p / np.sum(p)
        samples = np.random.choice(n1, n_negs, p=p)
        negs2.append(list(samples))

    negs1 = np.array(negs1, dtype=np.int64)
    negs2 = np.array(negs2, dtype=np.int64)

    return negs1, negs2


def negative_edge_sampling(prob, anchors, n_negs):
    anchors = anchors.cpu().detach().numpy()
    prob = prob[anchors]
    n = prob.shape[1]
    negs = []
    for p in prob:
        p1 = p / np.sum(p)
        neg_idx = np.random.choice(n, n_negs, p=p1)
        negs.append(neg_idx)

    negs = np.array(negs, dtype=np.int64)

    return negs
