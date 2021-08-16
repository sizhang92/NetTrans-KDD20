import torch
from torch_geometric.nn import GCNConv, GraphConv
from torch_sparse import coalesce
from functions.unpooling_aux_func import remove_by_threshold


class TransUnPool(torch.nn.Module):
    def __init__(self, in_channels, out_channels, non_linearity=torch.tanh):
        super(TransUnPool, self).__init__()
        self.in_channels = in_channels
        self.non_linearity = non_linearity
        self.out_channels = out_channels
        self.BiConv = GraphConv((in_channels, in_channels), out_channels, bias=False)
        self.UniConv = GCNConv(in_channels, out_channels, bias=False, improved=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.BiConv.reset_parameters()
        self.UniConv.reset_parameters()

    def forward(self, sup_x, y, edge_index, edge_weight, assign_index, assign_weight):
        num_nodes1 = y.shape[0]
        num_nodes2 = sup_x.shape[0]

        x = self.BiConv((sup_x, None), assign_index, assign_weight, size=(num_nodes2, num_nodes1))
        x += self.UniConv(y, edge_index, edge_weight)

        # edge_weight = torch.cat(
        #     [edge_weight, self.non_linearity(torch.sum(x[edge_index[0]] * x[edge_index[1]], dim=1).view(-1))],
        #     dim=0)
        #
        # edge_index = torch.cat([edge_index, edge_index], dim=1)
        # edge_index, edge_weight = coalesce(edge_index, 0.5 * edge_weight, num_nodes1, num_nodes1)
        # edge_index, edge_weight = remove_by_threshold(edge_index, edge_weight, 0.3)

        return x, edge_index, edge_weight


class LastLayer(torch.nn.Module):
    def __init__(self, in_channels1, out_channels, bias=False):
        super(LastLayer, self).__init__()
        self.in_channels1 = in_channels1    # bipartite graph emb_dim
        self.out_channels = out_channels
        self.BiConv1 = GraphConv((in_channels1, in_channels1), out_channels, bias=bias)
        self.BiConv2 = GraphConv((in_channels1, in_channels1), out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.BiConv1.reset_parameters()
        self.BiConv2.reset_parameters()

    def forward(self, sup_x, y, assign_index, assign_weight, anchor_links, anchor_weight, num_nodes):
        """"""
        # assign_index is of bipartite graph, edge_index is of unipartite graph
        # sup_x is embedding of supernodes, x0 is the node attribute of target network
        # y is the node embedding of source network
        num_src_nodes = y.shape[0]
        num_nodes1 = sup_x.shape[0]
        z = self.BiConv2((y, None), anchor_links, anchor_weight, size=(num_src_nodes, num_nodes))
        if assign_index is not None and assign_weight is not None:
            z += self.BiConv1((sup_x, None), assign_index, assign_weight, size=(num_nodes1, num_nodes))

        return z
