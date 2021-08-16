from torch_geometric.nn import GCNConv, GATConv, APPNP
from torch_geometric.nn.pool.topk_pool import topk
import torch
from torch.nn import Parameter
from functions.sparse_tensor_func import degree_normalize_sparse_tensor, sparse_gumbel_softmax
from functions.pooling_aux_func import filter_target, cadicate_selection, remap_sup, \
    filter_source, isolated_source_nodes, connect_isolated_nodes
from torch.sparse import mm
from functions.unpooling_aux_func import remove_by_threshold
from torch_geometric.utils import remove_self_loops
import math


class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, k=10, alpha=0.5):
        super(GCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.alpha = alpha

        self.conv = APPNP(self.k, self.alpha, add_self_loops=False)
        self.weight = torch.empty((self.in_channels, self.out_channels))
        torch.nn.init.uniform_(self.weight,  a=-math.sqrt(out_channels), b=math.sqrt(out_channels))
        self.weight = Parameter(self.weight)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight):
        x = x @ self.weight
        x = self.conv(x, edge_index, edge_weight)

        return x


class TransPool(torch.nn.Module):
    def __init__(self, in_channels, out_channels, ratio=0.85, non_linearity=torch.relu):
        super(TransPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.score_layer = GCNConv(in_channels, 1)  # scoring layer for self-attention pooling
        self.non_linearity = non_linearity
        self.supAggr_layer = GCNConv(in_channels, out_channels) # aggregation layer from nodes to supernodes
        self.gumbel_weight = torch.empty(in_channels, out_channels)
        torch.nn.init.xavier_uniform_(self.gumbel_weight)
        self.gumbel_weight = Parameter(self.gumbel_weight)
        self.reset_parameters()

    def reset_parameters(self):
        self.score_layer.reset_parameters()
        self.supAggr_layer.reset_parameters()


    def forward(self, x, edge_index, temperature, edge_weight, batch, layer_agg='avg'):
        num_nodes = x.shape[0]
        nodes = torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        # select super nodes
        score = self.score_layer(x, edge_index, edge_weight).reshape(-1)
        super_nodes = topk(score, self.ratio, batch)
        num_supernodes = len(super_nodes)
        batch = batch[super_nodes]

        # build bipartite graph for node-to-supernode aggregation
        edge_index1, edge_weight1 = filter_target(edge_index, edge_weight, super_nodes, num_nodes)
        if isinstance(self.supAggr_layer, GCNConv):
            temp_x = self.supAggr_layer(x, edge_index1, edge_weight1)
        elif isinstance(self.supAggr_layer, GATConv):
            temp_x = self.supAggr_layer(x, edge_index1)
        else:
            raise ValueError("Currently only GAT and GCN are supported for node-to-supernode aggregation.")

        sup_x = self.non_linearity(temp_x[super_nodes])
        # re-map the index of supernodes to [0, num_supernodes-1]
        edge_index1, _ = remap_sup(edge_index1, super_nodes, num_nodes)
        out_edge_index, out_edge_weight = filter_source(edge_index1, edge_weight1, super_nodes, num_nodes)
        out_edge_index, _ = remap_sup(out_edge_index, super_nodes, num_nodes, mode='source')

        # candidate selection
        edge_index, edge_weight = cadicate_selection(edge_index, edge_weight, edge_index1, edge_weight1,
                                                     nodes, super_nodes)

        assign_weight = torch.sigmoid(torch.sum(temp_x[edge_index[0]] * sup_x[edge_index[1]], dim=1))
        assign_weight = assign_weight * edge_weight
        edge_index, assign_weight = degree_normalize_sparse_tensor(edge_index, assign_weight, shape=[num_nodes, num_supernodes])
        edge_index, assign_weight = sparse_gumbel_softmax(assign_weight, edge_index, temperature, shape=[num_nodes, num_supernodes])
        edge_index, assign_weight = remove_by_threshold(edge_index, assign_weight, 0.5 / sup_x.shape[0])

        assign_index = torch.stack([edge_index[1], edge_index[0]])  # transpose
        if layer_agg == 'skip':
            x = sup_x
        else:
            assign = torch.sparse_coo_tensor(assign_index, assign_weight, size=[num_supernodes, num_nodes],
                                        device=edge_index.device)
            x = mm(assign, temp_x)
            if layer_agg == 'max':
                x = torch.max(sup_x, x)
            elif layer_agg == 'avg':
                x = (sup_x + x) / 2

        # get coarsened edges
        mapped_supernodes = torch.arange(len(super_nodes), dtype=torch.long, device=edge_index.device)
        isolated_nodes = isolated_source_nodes(out_edge_index, mapped_supernodes)
        if len(isolated_nodes) > 0:
            out_edge_index, out_edge_weight = connect_isolated_nodes(x, out_edge_index, out_edge_weight, isolated_nodes)

        out_edge_index, out_edge_weight = remove_self_loops(out_edge_index, out_edge_weight)

        return x, out_edge_index, out_edge_weight, batch, super_nodes, assign_index, assign_weight

