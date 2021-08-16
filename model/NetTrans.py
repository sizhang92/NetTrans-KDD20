import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn
from layers.TransPool import TransPool, GCN
from layers.TransUnPool import TransUnPool, LastLayer
from torch.sparse import mm
from torch_geometric.utils.repeat import repeat


class NetTrans(nn.Module):
    def __init__(self, args, act=nn.ReLU()):
        super(NetTrans, self).__init__()
        self.in_channels = args.in_channels
        self.nhid = args.nhid
        self.out_channels = args.out_channels
        self.pool_ratios = repeat(args.pooling_ratio, args.depth)
        self.depth = args.depth
        self.act = act
        self.down_convs = torch.nn.ModuleList()
        self.down_convs.append(GCN(self.in_channels, self.nhid, k=10))
        self.pools = torch.nn.ModuleList()
        # create encoder
        for i in range(args.depth):
            self.pools.append(TransPool(args.nhid, args.nhid, self.pool_ratios[i], non_linearity=self.act))
            self.down_convs.append(GCNConv(args.nhid, args.nhid))

        # create intermediate MLP
        self.MLP1 = nn.Linear(self.nhid, self.nhid, bias=True)

        # create decoder
        self.unpools = torch.nn.ModuleList()
        for i in range(args.depth - 1):
            self.unpools.append(TransUnPool(self.nhid, self.nhid))

        self.last_layer = LastLayer(self.nhid, self.nhid)

        self.MLP2 = torch.nn.Linear(self.nhid, self.out_channels)

        self.adj_loss_func = nn.BCEWithLogitsLoss()
        # self.align_loss_func = nn.BCEWithLogitsLoss()
        self.align_loss_func = nn.MarginRankingLoss(margin=args.margin)
        self.mse_loss = nn.MSELoss()
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for unpool in self.unpools:
            unpool.reset_parameters()
        self.MLP1.reset_parameters()
        self.MLP2.reset_parameters()
        self.last_layer.reset_parameters()

    def forward(self, x, edge_index, edge_weight, y, edge_index_y, edge_weight_y, anchor_links, temperature):
        """"""
        n1, n2 = x.shape[0], y.shape[0]
        batch = edge_index.new_zeros(x.shape[0])
        x = self.down_convs[0](x, edge_index, edge_weight)
        out_x = x

        x = self.act(x)

        xs, edge_indices, edge_weights, assign_mats = [out_x], [edge_index], [edge_weight], []
        num_sups = []

        for i in range(1, self.depth + 1):
            # TransPool layers to do the coarsening
            x, edge_index, edge_weight, batch, super_nodes, assign_index, assign_weight = self.pools[i - 1](
                x, edge_index, temperature, edge_weight, batch)
            assign_mats.append((assign_index, assign_weight))   # P matrices are sparse, so we store indices and values here
            num_sups.append(len(super_nodes))
            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)
            xs.append(x)
            if i < self.depth:
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
        sup1 = xs[1] if self.depth > 0 else []

        # intermediate layer
        x = self.act(self.MLP1(x))

        # decoder
        for i in range(self.depth - 1):
            j = self.depth - 1 - i
            x, edge_index, edge_weight = self.unpools[i](x, xs[j], edge_indices[j], edge_weights[j],
                                                         assign_mats[j][0], assign_mats[j][1])
            x = self.act(x)

        sup2 = x
        anchor_weight = torch.ones(len(anchor_links[0]), dtype=torch.float, device=x.device)
        q_index, q_weight = None, None
        if self.depth > 0:
            P1 = torch.sparse_coo_tensor(assign_mats[0][0], assign_mats[0][1], size=[num_sups[0], n1], device=x.device)
            L = torch.sparse_coo_tensor(anchor_links, anchor_weight, size=[n1, n2], device=x.device)
            Q = mm(P1, L).coalesce()
            q_index, q_weight = Q.indices(), Q.values()

        out_y = self.down_convs[0](y, edge_index_y, edge_weight_y)
        out_y += self.last_layer(sup2, xs[0], q_index, q_weight, anchor_links, anchor_weight, n2)

        recon_y = self.MLP2(out_y)

        return out_x, sup1, out_y, sup2, assign_mats, recon_y

    def score(self, emb1, emb2):
        score = -torch.sum(torch.abs(emb1 - emb2), dim=1).reshape((-1, 1))
        return score

    def adj_loss(self, anchor1_emb, context_pos1_emb, context_neg1_emb):

        num_instance1 = anchor1_emb.shape[0]
        num_instance2 = context_neg1_emb.shape[0]
        N_negs = num_instance2 // num_instance1
        dim = anchor1_emb.shape[1]
        device = anchor1_emb.device

        term1 = self.score(anchor1_emb, context_pos1_emb)
        term2 = self.score(anchor1_emb.repeat(1, N_negs).reshape(-1, dim), context_neg1_emb)

        terms1 = torch.cat([term1, term2], dim=0).reshape((-1,))
        labels1 = torch.cat([torch.ones(num_instance1, device=device), torch.zeros(num_instance2, device=device)])
        loss = self.adj_loss_func(terms1, labels1)

        return loss

    def align_loss(self, pos_emb1, pos_emb2, neg_emb1, neg_emb2):
        num_instance1 = pos_emb1.shape[0]
        num_instance2 = neg_emb1.shape[0]
        dim = pos_emb1.shape[1]
        device = pos_emb1.device
        N_negs = num_instance2 // num_instance1
        term1 = self.score(pos_emb1, pos_emb2)
        term2 = self.score(pos_emb2, pos_emb1)
        term3 = self.score(pos_emb1.repeat(1, N_negs).reshape(-1, dim), neg_emb1)
        term4 = self.score(pos_emb2.repeat(1, N_negs).reshape(-1, dim), neg_emb2)
        # labels1 = torch.cat([torch.ones(num_instance1, device=device), torch.zeros(num_instance2, device=device)])
        # labels2 = torch.cat([torch.ones(num_instance1, device=device), torch.zeros(num_instance2, device=device)])
        # terms1 = torch.cat([term1, term3], dim=0).reshape((-1,))
        # terms2 = torch.cat([term2, term4], dim=0).reshape((-1,))
        loss = self.align_loss_func(term1.repeat(1, N_negs).reshape((-1,)), term3, torch.ones_like(term3)) + \
                self.align_loss_func(term2.repeat(1, N_negs).reshape((-1,)), term4, torch.ones_like(term4))
        # loss = self.align_loss_func(terms1, labels1) + self.align_loss_func(terms2, labels2)

        return loss

    def recon_attr_loss(self, recon_y, y):
        return self.mse_loss(y, recon_y)

