import torch
from torch_geometric.utils import remove_self_loops, segregate_self_loops
from torch.sparse import mm
from torch_geometric.utils import add_self_loops


def cadicate_selection(edge_index, edge_weight, edge_index1, edge_weight1, nodes, supernodes):
    '''
    Select candidate supernodes for nodes to be merged into.

    :param edge_index: edge indices of the original input graph
    :param edge_weight: edge weights of the original input graph
    :param edge_index1: edge indices of filtered node-to-supernode graph
    :param edge_weight1: edge weights of filtered node-to-supernode graph
    :param nodes: nodes of the input graph
    :param supernodes: supernodes of input graph
    :return:
        edge_index: candidate connections after augmentation between nodes and supernodes
        edge_weight: connection weights after augmentation between nodes and supernodes
    '''

    num_nodes, num_supernodes = len(nodes), len(supernodes)
    # detect if isolated nodes exist which are not connected with supernodes within 1-hop
    isolated_nodes = isolated_source_nodes(edge_index1, nodes)
    num_isolated = len(isolated_nodes)
    # expand a 2-hop graph with respect to the isolated nodes if existed
    if num_isolated > 0:
        edge_index_iso, edge_weight_iso = filter_source(edge_index, edge_weight, isolated_nodes, num_nodes)
        # edge_index_iso, _ = remap_sup(edge_index_iso, isolated_nodes, num_nodes, mode='source')
        edge_index_iso, edge_weight_iso = two_hop_graph(edge_index_iso, edge_weight_iso, [num_nodes, num_nodes],
                                                        edge_index1, edge_weight1, [num_nodes, num_supernodes])
        edge_index1 = torch.cat([edge_index1, edge_index_iso], dim=1)
        edge_weight1 = torch.cat([edge_weight1, edge_weight_iso], dim=0)
        # check if isolated nodes still exist (i.e., not connected to other nodes within two hops)
        # if so, connect them to supernodes with uniform probabilities
        isolated_nodes = isolated_source_nodes(edge_index1, nodes)
        num_isolated = len(isolated_nodes)
        if num_isolated > 0:
            edge_index1, edge_weight1 = augment_uniformly(edge_index1, edge_weight1, isolated_nodes, num_supernodes)

    edge_index, edge_weight = preserve_loops(edge_index1, edge_weight1, supernodes, num_nodes)

    return edge_index, edge_weight


def preserve_loops(edge_index, edge_weight, supernodes, num_nodes):
    '''
    For nodes that are selected as supernodes, only keep one nonzero element in node-to-supernode assignment
    matrix such that they are assigned strictly to themselves.

    :param edge_index: node-to-supernode assignments
    :param edge_weight: node-to-supernode assignment weight
    :param supernodes: nodes that are selected as supernodes
    :param num_nodes: number of nodes in the original graph
    :return:
        edge_index: assignment indices with non-related entries removed
        edge_weight: corresponding assignment weights
    '''

    mask = torch.zeros(num_nodes, device=edge_index.device)
    mask[supernodes] = 1

    row, col = edge_index
    mask = (mask[row] == 0)
    row, col = row[mask], col[mask]
    edge_weight = edge_weight[mask]
    edge_index = torch.stack([row, col])

    self_loops = torch.stack([supernodes, torch.arange(len(supernodes), dtype=torch.long, device=supernodes.device)])
    self_weight = torch.ones(len(supernodes), dtype=torch.float, device=supernodes.device)

    edge_index = torch.cat([edge_index, self_loops], dim=1)
    edge_weight = torch.cat([edge_weight, self_weight], dim=0)

    return edge_index, edge_weight


def filter_source(edge_index, edge_weight, source_nodes, num_nodes):
    '''
    Filter out the edges with source nodes
    :param edge_index: edge indices of the input graph
    :param edge_weight: edge weights of the input graph
    :param source_nodes: source nodes to be filtered
    :param num_nodes: number of nodes of the input graph

    :return:
        edge_index: edge indices of the filtered graph
        edge_weight: edge weights of the filtered graph
    '''
    mask = torch.zeros(num_nodes, device=edge_index.device)
    mask[source_nodes] = 1

    row, col = edge_index
    mask = (mask[row] > 0)
    row, col = row[mask], col[mask]
    edge_weight = edge_weight[mask]
    edge_index = torch.stack([row, col])

    return edge_index, edge_weight


def filter_target(edge_index, edge_weight, target_nodes, num_nodes):
    '''
    Extract a directed self-looped sub-bipartite graph from the original graph (i.e., edge_index)
    based on the selected supernodes

    :param edge_index: edge indices of the input graph
    :param edge_weight: edge weights of the input graph
    :param supernodes: selected super nodes by self-attention pooling
    :param num_nodes: number of nodes of input graph

    :return:
        edge_index: edge indices of the extracted directed bipartite graph
        edge_weight: edge weights of the extracted directed bipartite graph
    '''
    mask = torch.zeros(num_nodes, device=edge_index.device)
    mask[target_nodes] = 1

    # # add self loops to enforce nodes that are selected as supernodes can be merged to themselves
    # edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1000)

    row, col = edge_index
    mask = (mask[col] > 0)
    row, col = row[mask], col[mask]
    edge_weight = edge_weight[mask]
    edge_index = torch.stack([row, col])

    return edge_index, edge_weight


def remap_sup(edge_index, super_nodes, num_nodes, mode='target'):
    '''
    Remap supernode indices to [0, num_supernodes - 1].

    :param edge_index: edge indices of input node-to-supernode graph
    :param super_nodes: supernode indices
    :param num_nodes: number of nodes

    :return:
        edge_index: edge indices with remapped supernodes
        index_map: node mapping
    '''
    index_map = -1 * torch.ones(num_nodes, dtype=torch.long, device=edge_index.device)
    index_map[super_nodes] = torch.arange(len(super_nodes), dtype=torch.long, device=edge_index.device)
    if mode == 'target':
        row, col = edge_index
        col = index_map[col]
        if -1 in col:
            raise ValueError("Invalid index in the mapping.")
    else:
        row, col = edge_index
        row = index_map[row]
        if -1 in row:
            raise ValueError("Invalid index in the mapping.")

    edge_index = torch.stack([row, col])
    return edge_index, index_map


def isolated_source_nodes(edge_index, nodes):
    '''
    Detect isolated source nodes.

    :param edge_index: edge indices of the input graph
    :param nodes: all nodes of the input graph

    :return:
        isolated_idx: indices of isolated source nodes
    '''

    mask = torch.ones(max(nodes) + 1, dtype=torch.bool, device=edge_index.device)
    mask[edge_index[0]] = False
    mask = mask[nodes]
    isolated_idx = nodes[mask]

    return isolated_idx


def two_hop_graph(edge_index, edge_weight, graph_shape,
                  edge_index1, edge_weight1, graph_shape1):
    '''
    Return graph whose nodes are connected by 2-hops originally

    :param edge_index: edge indices of input graph
    :param edge_weight: edge weights of input graph
    :param num_nodes: number of nodes of input graph
    :param edge_index1: edge indices of another input graph
    :param edge_weight1: edge weights of another input graph
    :param num_nodes1: number of nodes of another input graph

    :return:
        edge_index: edge indices of the 2-hop graph
        edge_weight: edge weights of the 2-hop graph
    '''

    A = torch.sparse_coo_tensor(edge_index, edge_weight, size=graph_shape, device=edge_index.device)
    B = torch.sparse_coo_tensor(edge_index1, edge_weight1, size=graph_shape1, device=edge_index.device)

    A = mm(A, B).coalesce()
    edge_index = A.indices()
    edge_weight = A.values()

    return edge_index, edge_weight


def augment_uniformly(edge_index, edge_weight, nodes, num_supernodes):
    '''
    Connect isolated nodes to supernodes with uniform weights

    :param edge_index: edge indices of given node-to-supernode indices
    :param edge_weight: edge weights of given node-to-supernode indices
    :param nodes: nodes of input graph
    :param supernodes: supernodes of input graph
    :param index_map: mapping of supernodes
    :return:
        edge_index: augmented edge indices
        edge_weight: augmented edge weights
    '''
    num_nodes = len(nodes)
    nodes = nodes.reshape(-1, 1).repeat(1, num_supernodes).reshape(-1)
    supernodes = torch.arange(num_supernodes, dtype=torch.long, device=edge_index.device)
    supernodes = supernodes.reshape(1, -1).repeat(num_nodes, 1).reshape(-1)

    augment_index = torch.stack([nodes, supernodes])
    augment_weight = (1/num_supernodes) * torch.ones(len(supernodes), dtype=torch.float, device=edge_weight.device)

    edge_index = torch.cat([edge_index, augment_index], dim=1)
    edge_weight = torch.cat([edge_weight, augment_weight], dim=0)

    return edge_index, edge_weight


def connect_isolated_nodes(x, edge_index, edge_weight, isolated_nodes):

    out = segregate_self_loops(edge_index, edge_weight)
    edge_index, edge_attr, loop_edge_index, loop_edge_weight = out
    x_isolated = x[isolated_nodes]
    aux_weights = torch.sigmoid(x_isolated.mm(x.t())).view(-1)

    num_nodes = x.shape[0]
    nodes = torch.arange(num_nodes, dtype=torch.long, device=x.device)
    original_idx = nodes.view(1, -1).repeat(len(isolated_nodes), 1).view(1, -1)
    isolated_idx = isolated_nodes.view(-1, 1).repeat(1, num_nodes).view(1, -1)
    aux_index = torch.cat([isolated_idx, original_idx], dim=0)
    aux_index1 = torch.cat([original_idx, isolated_idx], dim=0)

    # to undirected
    aux_index = torch.cat([aux_index, aux_index1], dim=1)
    aux_weights = torch.cat([aux_weights, aux_weights], dim=0)
    aux_index, aux_weights = remove_self_loops(aux_index, aux_weights)

    edge_index = torch.cat([edge_index, aux_index], dim=1)
    edge_weight = torch.cat([edge_weight, aux_weights], dim=0)

    return edge_index, edge_weight