import argparse
import numpy as np
import torch
import time
from utils.data import Train_Data
from torch.utils.data import DataLoader
from model.NetTrans import NetTrans
from model.negative_sampling import negative_edge_sampling, negative_sampling_uniform
from utils.util import load_data, get_neighbors, neg_context_prob, balance_inputs, anchor_as_attr
from sklearn.metrics.pairwise import manhattan_distances


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=123, help='seed')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--ratio', type=float, default=0.2, help='training ratio')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--depth', type=int, default=2, help='depth of encoder')
parser.add_argument('--gpu', type=int, default=0, help='gpu number.')
parser.add_argument('--pooling_ratio', type=float, default=0.2, help='pooling ratio')
parser.add_argument('--attr_coeff', type=float, default=1., help='weight for reconstructing attributes')
parser.add_argument('--adj_coeff', type=float, default=1., help='weight for reconstructing adjacency')
parser.add_argument('--rank_coeff', type=float, default=1., help='weight for ranking loss')
parser.add_argument('--margin', type=float, default=1., help='margin for ranking loss.')
parser.add_argument('--dataset', type=str, default='foursquare-twitter', help='dataset name')
parser.add_argument('--epochs', type=int, default=50, help='maximum number of epochs')
parser.add_argument('--neg_size', type=int, default=20, help='negative sample size')
parser.add_argument('--batch_size', type=int, default=300, help='batch_size')


args = parser.parse_args()
edge_index1, edge_index2, x1, x2, anchor_links, test_pairs = load_data('datasets/' + args.dataset, args.ratio)
x1, x2 = anchor_as_attr(x1, x2, anchor_links)
n1, args.in_channels = x1.shape
n2, args.out_channels = x2.shape

# get context pairs from neighboring nodes
context_pairs1 = get_neighbors(edge_index1, anchor_links[0])
context_pairs2 = get_neighbors(edge_index2, anchor_links[1])
context_pairs1, context_pairs2 = balance_inputs(context_pairs1, context_pairs2)
neg_context_prob1, anchor_map1 = neg_context_prob(edge_index1, anchor_links[0], n1)
neg_context_prob2, anchor_map2 = neg_context_prob(edge_index2, anchor_links[1], n2)

dataset = Train_Data(context_pairs1, context_pairs2)
data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)


args.device = 'cpu'
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:%d' % args.gpu

# to cuda
edge_index1 = torch.from_numpy(edge_index1).to(args.device)
edge_index2 = torch.from_numpy(edge_index2).to(args.device)
x1 = torch.from_numpy(x1).to(args.device)
x2 = torch.from_numpy(x2).to(args.device)

anchors = torch.from_numpy(anchor_links).to(args.device)
anchor_map1 = torch.from_numpy(anchor_map1).to(args.device)
anchor_map2 = torch.from_numpy(anchor_map2).to(args.device)

edge_weight1 = torch.ones(len(edge_index1[0]), dtype=torch.float, device=args.device)
edge_weight2 = torch.ones(len(edge_index2[0]), dtype=torch.float, device=args.device)

model = NetTrans(args).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

ANNEAL_RATE = 0.00002
temp_min = 0.1
temp = 1


def test(model, x, y, test_set):
    model.eval()
    metric = [1, 10, 30, 50, 100]

    test_nodes1, test_nodes2 = test_set[0], test_set[1]
    with torch.no_grad():
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        dist1 = manhattan_distances(x[test_nodes1], y)
        dist2 = manhattan_distances(y[test_nodes2], x)

        idx1 = np.argsort(dist1, axis=1)[:, :100]
        idx2 = np.argsort(dist2, axis=1)[:, :100]
        test_set = set(tuple(i) for i in test_set.T)
        hits_l = []
        for k in metric:
            id2 = idx1[:, :k].reshape((-1, 1))
            idx = np.repeat(test_nodes1.reshape((-1, 1)), k, axis=1).reshape(-1, 1)
            idx = np.concatenate([idx, id2], axis=1)
            idx = set(tuple(i) for i in idx)
            count = len(idx.intersection(test_set))
            hit = count/len(test_set)
            hits_l.append(hit)

        hits_r = []
        for k in metric:
            id2 = idx2[:, :k].reshape((-1, 1))
            idx = np.repeat(test_nodes2.reshape((1, -1)), k, axis=0).reshape((-1, 1))
            idx = np.concatenate([id2, idx], axis=1)
            idx = set(tuple(i) for i in idx)
            count = len(idx.intersection(test_set))
            hit = count / len(test_set)
            hits_r.append(hit)

        hits_l = np.array(hits_l)
        hits_r = np.array(hits_r)
        hits = np.maximum(hits_l, hits_r)

    return hits


t_neg_sampling, t_model, t_loss = 0, 0, 0
max_hits = np.zeros(5, dtype=np.float32)
train_hits = []
hits = []
max_hit_30 = 0
max_epoch = 0

print('start training.')
for epoch in range(args.epochs):
    model.train()
    if epoch % 10 == 1:
        temp = np.maximum(temp * np.exp(-ANNEAL_RATE * epoch), temp_min)
    for i, data in enumerate(data_loader):
        nodes1, nodes2 = data
        nodes1 = nodes1.to(args.device)
        nodes2 = nodes2.to(args.device)

        anchor_nodes1 = nodes1[:, 0].reshape(-1)
        pos_context_nodes1 = nodes1[:, 1].reshape(-1)
        anchor_nodes2 = nodes2[:, 0].reshape(-1)
        pos_context_nodes2 = nodes2[:, 1].reshape(-1)

        optimizer.zero_grad()
        for name, param in model.named_parameters():
            if param.isnan().any():
                print(name)
        t0 = time.time()
        x, sup1, y, sup2, assign_mats, recon_y = model(x1, edge_index1, edge_weight1,
                                                       x2, edge_index2, edge_weight2, anchors, temp)
        t_model += (time.time() - t0)

        # negative sampling
        t0 = time.time()
        negs1, negs2 = negative_sampling_uniform(anchor_nodes1, anchor_nodes2, n1, n2, args.neg_size)
        neg_context1 = negative_edge_sampling(neg_context_prob1, anchor_map1[anchor_nodes1], args.neg_size)
        neg_context2 = negative_edge_sampling(neg_context_prob2, anchor_map2[anchor_nodes2], args.neg_size)

        negs1 = torch.from_numpy(negs1).reshape(-1).to(args.device)
        negs2 = torch.from_numpy(negs2).reshape(-1).to(args.device)
        neg_context1 = torch.from_numpy(neg_context1).reshape(-1).to(args.device)
        neg_context2 = torch.from_numpy(neg_context2).reshape(-1).to(args.device)

        t_neg_sampling += (time.time() - t0)

        t0 = time.time()

        neg_emb1, neg_emb2 = y[negs1], x[negs2]
        neg_context_emb1, neg_context_emb2 = x[neg_context1], y[neg_context2]
        anchor_emb1, anchor_emb2 = x[anchor_nodes1], y[anchor_nodes2]
        pos_emb1, pos_emb2 = x[pos_context_nodes1], y[pos_context_nodes2]

        adj_loss = model.adj_loss(anchor_emb1, pos_emb1, neg_context_emb1) + \
                   model.adj_loss(anchor_emb2, pos_emb2, neg_context_emb2)
        align_loss = model.align_loss(anchor_emb1, anchor_emb2, neg_emb1, neg_emb2)
        total_loss = args.adj_coeff * adj_loss + args.rank_coeff * align_loss

        t_loss += (time.time() - t0)

        print("Epoch:{}, Batch:{}/{}, Training loss:{}".format(
            epoch + 1, i + 1, len(data_loader), round(total_loss.item(), 4)))

        total_loss.backward()
        optimizer.step()

    train_hits = test(model, x, y, anchor_links)
    hits = test(model, x, y, test_pairs)
    print("Epoch:{}, Train_Hits:{}, Hits:{}".format(epoch + 1, np.round(train_hits, 4), np.round(hits, 4)))
    # max_hits = np.maximum(max_hits, np.array(hits))
    if hits[2] > max_hit_30:
        max_hits = hits
        max_hit_30 = hits[2]
        max_epoch = epoch + 1

    print(max_hits, max_epoch)

with open('results/results_%s_%.1f.txt' % (args.dataset, args.ratio), 'a+') as f:
    f.write(', '.join([str(x) for x in np.round(max_hits, 4)]) + '\n')

