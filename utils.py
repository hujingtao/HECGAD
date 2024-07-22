import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.io as sio
import random
import torch
import dgl
import os
import pickle
from torch_geometric.utils import to_scipy_sparse_matrix, to_undirected
from collections import OrderedDict

def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

# copy from inject_anomaly.py
def dense_to_sparse(dense_matrix):
    shape = dense_matrix.shape
    row = []
    col = []
    data = []
    for i, r in enumerate(dense_matrix):
        for j in np.where(r > 0)[0]:
            row.append(i)
            col.append(j)
            data.append(dense_matrix[i, j])

    sparse_matrix = sp.coo_matrix((data, (row, col)), shape=shape).tocsc()
    return sparse_matrix


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot



def load_pt(dataset):
    file_path = os.path.join('./data/', dataset + '.pt')
    pygdata = torch.load(file_path)
    edge = to_undirected(pygdata.edge_index)
    network = to_scipy_sparse_matrix(edge)
    adj = sp.csr_matrix(network)
    feat = pygdata.x.numpy()
    features = dense_to_sparse(feat)
    ano_labels = pygdata.y.numpy()
    return adj, features, ano_labels


def load_dat(dataset):
    pygdata = pickle.load(open('./data/{}.dat'.format(dataset), 'rb'))
    edge = to_undirected(pygdata.edge_index)
    network = to_scipy_sparse_matrix(edge)
    adj = sp.csr_matrix(network)
    feat = pygdata.x.numpy()
    ano_labels = pygdata.y.numpy()
    return adj, feat, ano_labels



def load_mat(dataset, train_rate=0.3, val_rate=0.1):
    """Load .mat dataset."""
    data = sio.loadmat("./dataset/{}.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']

    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)
    # labels = np.squeeze(np.array(data['Class'], dtype=np.int64) - 1)
    # num_classes = np.max(labels) + 1
    # labels = dense_to_one_hot(labels, num_classes)
    ano_labels = np.squeeze(np.array(label))

    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[: num_train]
    idx_val = all_idx[num_train: num_train + num_val]
    idx_test = all_idx[num_train + num_val:]

    # return adj, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels
    return adj, feat, ano_labels


def load_mat2(dataset, train_rate=0.3, val_rate=0.1):
    """Load .mat dataset."""
    data = sio.loadmat("./dataset/{}.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']

    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)
    labels = np.squeeze(np.array(data['Class'], dtype=np.int64) - 1)
    # num_classes = np.max(labels) + 1
    # labels = dense_to_one_hot(labels, num_classes)
    ano_labels = np.squeeze(np.array(label))

    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[: num_train]
    idx_val = all_idx[num_train: num_train + num_val]
    idx_test = all_idx[num_train + num_val:]

    # return adj, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels
    return adj, feat, labels, ano_labels



def load_single_ano_mat(ano_type, dataset):
    """Load .mat dataset."""
    if ano_type == 'attr':
        path = 'Attribute'
    else:
        path = 'Topology'
    data = sio.loadmat("./Single_Anomaly_Data/{}/{}.mat".format(path, dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']

    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)
    ano_labels = np.squeeze(np.array(label))

    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
    else:
        str_ano_labels = None

    if 'attr_anomaly_label' in data:
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        attr_ano_labels = None

    # return adj, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels
    return adj, feat, ano_labels, str_ano_labels, attr_ano_labels



def adj_to_dgl_graph(adj):
    """Convert adjacency matrix to dgl format."""
    nx_graph = nx.from_scipy_sparse_matrix(adj)
    dgl_graph = dgl.DGLGraph(nx_graph)
    return dgl_graph


def get_first_adj(dgl_graph, adj, subgraph_size):
    """Generate the first view's subgraph with the first-order neighbor."""
    subgraphs = []
    all_idx = list(range(dgl_graph.number_of_nodes()))

    adj = np.array(adj.todense())
    row, col = np.diag_indices_from(adj)
    zeros = np.zeros(adj.shape[0])
    adj[row, col] = np.array(zeros)
    adj = adj.squeeze()

    for node_id in all_idx:
        first_adj = np.where(adj[node_id] == 1)
        first_adj = list(first_adj[0])
        if len(first_adj) < subgraph_size - 1:
            subgraphs.append(first_adj)
            first_adj_add = first_adj.copy()
            first_adj_add.append(node_id)
            subgraphs[node_id].extend(list(np.random.choice(first_adj_add, subgraph_size - len(first_adj) - 1, replace=True)))
        else:
            subgraphs.append(list(np.random.choice(first_adj, subgraph_size - 1, replace=False)))
        subgraphs[node_id].append(node_id)
    return subgraphs


def get_first_adj_old(dgl_graph, adj, subgraph_size):
    """Generate the first view's subgraph with the first-order neighbor."""
    all_idx = list(range(dgl_graph.number_of_nodes()))
    subgraphs = []
    adj = np.array(adj.todense()).squeeze()
    for node_id in all_idx:
        first_adj = np.where(adj[node_id] == 1)
        first_adj = list(first_adj[0])
        if len(first_adj) < subgraph_size - 1:
            subgraphs.append(first_adj)
            first_adj.append(node_id) #自己也可以被循环选择
            subgraphs[node_id].extend(
                list(np.random.choice(first_adj, subgraph_size - len(first_adj) - 1, replace=True)))
        else:
            subgraphs.append(list(np.random.choice(first_adj, subgraph_size - 1, replace=False)))
        subgraphs[node_id].append(node_id)
    return subgraphs



def get_first_adj_high(dgl_graph, A, subgraph_size, degree, avg_degree):
    """Generate the first view's subgraph with the first-order neighbor."""
    # node_num = A.shape[0]
    # degree = np.array(np.sum(A, axis=0)).squeeze()
    # total_edge = np.sum(A)
    # avg_degree = total_edge / node_num
    all_idx = list(range(dgl_graph.number_of_nodes()))
    subgraphs = []
    adj = np.array(A.todense()).squeeze()
    for node_id in all_idx:
        first_adj = np.where(adj[node_id] == 1)
        first_adj = list(first_adj[0])

        first_adj_high_idx = np.where(degree[first_adj] > avg_degree)[0]
        if len(first_adj_high_idx) > 0:
            first_adj = np.array(first_adj)
            first_adj = first_adj[list(first_adj_high_idx)]
            first_adj = list(first_adj)


        if len(first_adj) < subgraph_size - 1:
            subgraphs.append(first_adj)
            first_adj.append(node_id) #自己也可以被循环选择
            subgraphs[node_id].extend(
                list(np.random.choice(first_adj, subgraph_size - len(first_adj) - 1, replace=True)))
        else:
            subgraphs.append(list(np.random.choice(first_adj, subgraph_size - 1, replace=False)))
        subgraphs[node_id].append(node_id)
    return subgraphs



def get_first_adj_small(dgl_graph, A, subgraph_size, degree, avg_degree):
    """Generate the first view's subgraph with the first-order neighbor."""
    # node_num = A.shape[0]
    # degree = np.array(np.sum(A, axis=0)).squeeze()
    # total_edge = np.sum(A)
    # avg_degree = total_edge / node_num
    all_idx = list(range(dgl_graph.number_of_nodes()))
    subgraphs = []
    adj = np.array(A.todense()).squeeze()
    for node_id in all_idx:
        first_adj = np.where(adj[node_id] == 1)
        first_adj = list(first_adj[0])

        first_adj_small_idx = np.where(degree[first_adj] < avg_degree)[0]
        if len(first_adj_small_idx) > 0:
            first_adj = np.array(first_adj)
            first_adj = first_adj[list(first_adj_small_idx)]
            first_adj = list(first_adj)

        if len(first_adj) < subgraph_size - 1:
            subgraphs.append(first_adj)
            first_adj.append(node_id) #自己也可以被循环选择
            subgraphs[node_id].extend(
                list(np.random.choice(first_adj, subgraph_size - len(first_adj) - 1, replace=True)))
        else:
            subgraphs.append(list(np.random.choice(first_adj, subgraph_size - 1, replace=False)))
        subgraphs[node_id].append(node_id)
    return subgraphs



def get_second_adj_1(dgl_graph, adj, subgraph_size):
    """Generate the second view's subgraph with the first-order and second-order neighbor. Prior first-order."""
    all_idx = list(range(dgl_graph.number_of_nodes()))
    subgraphs = []
    adj_2 = adj.dot(adj)
    adj = np.array(adj.todense())
    adj_2 = np.array(adj_2.todense())
    row, col = np.diag_indices_from(adj_2)
    zeros = np.zeros(adj_2.shape[0])
    adj_2[row, col] = np.array(zeros)
    adj = adj.squeeze()
    adj_2 = adj_2.squeeze()
    for node_id in all_idx:
        first_adj = np.where(adj[node_id] == 1)
        second_adj = np.where(adj_2[node_id] != 0)
        first_adj = first_adj[0].tolist()
        second_adj = second_adj[0].tolist()
        if len(first_adj) < subgraph_size - 1:
            if len(second_adj) == 0:
                subgraphs.append(first_adj)
                first_adj.append(node_id)
                subgraphs[node_id].extend(
                    list(np.random.choice(first_adj, subgraph_size - len(first_adj) - 1, replace=True)))
            elif len(second_adj) + len(first_adj) < subgraph_size - 1:
                subgraphs.append(first_adj + second_adj)
                subgraphs[node_id].extend(
                    list(np.random.choice(second_adj, subgraph_size - len(first_adj) - len(second_adj) - 1,
                                          replace=True)))
            else:
                subgraphs.append(first_adj)
                subgraphs[node_id].extend(
                    list(np.random.choice(second_adj, subgraph_size - len(first_adj) - 1, replace=False)))
        else:
            subgraphs.append(list(np.random.choice(first_adj, subgraph_size // 2, replace=False)))
            if len(second_adj) == 0:
                subgraphs[node_id].extend(list(np.random.choice(first_adj, subgraph_size // 2 - 1, replace=False)))
            elif len(second_adj) < subgraph_size // 2 - 1:
                subgraphs[node_id].extend(list(np.random.choice(second_adj, subgraph_size // 2 - 1, replace=True)))
            else:
                subgraphs[node_id].extend(list(np.random.choice(second_adj, subgraph_size // 2 - 1, replace=False)))
        subgraphs[node_id].append(node_id)
    return subgraphs



def get_second_adj_2(dgl_graph, adj, subgraph_size):
    """Generate the second view's subgraph with the second-order neighbor. Prior second-order. If no second-order, completed with first-order."""
    all_idx = list(range(dgl_graph.number_of_nodes()))
    subgraphs = []
    adj_2 = adj.dot(adj)
    adj = np.array(adj.todense())
    adj_2 = np.array(adj_2.todense())
    row, col = np.diag_indices_from(adj_2)
    zeros = np.zeros(adj_2.shape[0])
    adj_2[row, col] = np.array(zeros)
    adj = adj.squeeze()
    adj_2 = adj_2.squeeze()
    for node_id in all_idx:
        first_adj = np.where(adj[node_id] == 1)
        second_adj = np.where(adj_2[node_id] != 0)
        first_adj = first_adj[0].tolist()
        second_adj = second_adj[0].tolist()

        if len(second_adj) == 0:
            if len(first_adj) < subgraph_size - 1:
                subgraphs.append(first_adj)
                first_adj.append(node_id)
                subgraphs[node_id].extend(list(np.random.choice(first_adj, subgraph_size - len(first_adj) - 1, replace=True)))
            else:
                subgraphs.append(list(np.random.choice(first_adj, subgraph_size - 1, replace=False)))
        elif len(second_adj) < subgraph_size - 1:
            if len(second_adj) + len(first_adj) < subgraph_size - 1:
                subgraphs.append(first_adj + second_adj)
                subgraphs[node_id].extend(list(np.random.choice(second_adj, subgraph_size - len(first_adj) - len(second_adj) - 1, replace=True)))
            else:
                subgraphs.append(second_adj)
                if len(first_adj) < subgraph_size - len(second_adj) - 1:
                    subgraphs[node_id].extend(list(np.random.choice(first_adj, subgraph_size - len(second_adj) - 1, replace=True)))
                else:
                    subgraphs[node_id].extend(list(np.random.choice(first_adj, subgraph_size - len(second_adj) - 1, replace=False)))
        else:
            subgraphs.append(list(np.random.choice(second_adj, subgraph_size - 1, replace=False)))

        subgraphs[node_id].append(node_id)
    return subgraphs


def get_second_adj_3(dgl_graph, adj, subgraph_size):
    """Generate the second view's subgraph with the 1/2 first-order and 1/2 second-order neighbor. """
    all_idx = list(range(dgl_graph.number_of_nodes()))
    subgraphs = []
    adj_2 = adj.dot(adj)
    adj = np.array(adj.todense())
    adj_2 = np.array(adj_2.todense())
    row, col = np.diag_indices_from(adj_2)
    zeros = np.zeros(adj_2.shape[0])
    adj_2[row, col] = np.array(zeros)
    adj = adj.squeeze()
    adj_2 = adj_2.squeeze()
    for node_id in all_idx:
        first_adj = np.where(adj[node_id] == 1)
        second_adj = np.where(adj_2[node_id] != 0)
        first_adj = first_adj[0].tolist()
        second_adj = second_adj[0].tolist()
        if len(first_adj) < subgraph_size // 2:
            subgraphs.append(list(np.random.choice(first_adj, subgraph_size // 2, replace=True)))
            if len(second_adj) == 0:
                first_adj.append(node_id)
                subgraphs[node_id].extend(list(np.random.choice(first_adj, (subgraph_size - 1) // 2, replace=True)))
            elif len(second_adj) < (subgraph_size - 1) // 2:
                subgraphs[node_id].extend(list(np.random.choice(second_adj, (subgraph_size - 1) // 2, replace=True)))
            else:
                subgraphs[node_id].extend(list(np.random.choice(second_adj, (subgraph_size - 1) // 2, replace=False)))
        else:
            if len(second_adj) == 0:
                first_adj.append(node_id)
                if len(first_adj) < subgraph_size - 1:
                    subgraphs.append(list(np.random.choice(first_adj, (subgraph_size - 1), replace=True)))
                else:
                    subgraphs.append(list(np.random.choice(first_adj, (subgraph_size - 1), replace=False)))
            elif len(second_adj) < (subgraph_size - 1) // 2 :
                subgraphs.append(list(np.random.choice(first_adj, subgraph_size // 2, replace=False)))
                subgraphs[node_id].extend(list(np.random.choice(second_adj, (subgraph_size - 1) // 2, replace=True)))
            else:
                subgraphs.append(list(np.random.choice(first_adj, subgraph_size // 2, replace=False)))
                subgraphs[node_id].extend(list(np.random.choice(second_adj, (subgraph_size - 1) // 2, replace=False)))

        subgraphs[node_id].append(node_id)
    return subgraphs



def get_second_adj_4(dgl_graph, adj, subgraph_size):
    """Generate the second view's subgraph with the only second-order neighbor. If no second-order, completed with itself."""
    all_idx = list(range(dgl_graph.number_of_nodes()))
    subgraphs = []
    adj_2 = adj.dot(adj)
    adj = np.array(adj.todense())
    adj_2 = np.array(adj_2.todense())
    row, col = np.diag_indices_from(adj_2)
    zeros = np.zeros(adj_2.shape[0])
    adj_2[row, col] = np.array(zeros)
    adj = adj.squeeze()
    adj_2 = adj_2.squeeze()
    for node_id in all_idx:
        second_adj = np.where(adj_2[node_id] != 0)
        second_adj = second_adj[0].tolist()

        if len(second_adj) == 0:
            second_adj.append(node_id)
            subgraphs.append(list(np.random.choice(second_adj, subgraph_size - 1, replace=True)))

        elif len(second_adj) < subgraph_size - 1:
            subgraphs.append(second_adj)
            second_adj.append(node_id)
            subgraphs[node_id].extend(list(np.random.choice(second_adj, subgraph_size - len(second_adj) - 1, replace=True)))

        else:
            subgraphs.append(list(np.random.choice(second_adj, subgraph_size - 1, replace=False)))

        subgraphs[node_id].append(node_id)
    return subgraphs


def find_1to2_neigh(target_node, subgraph, adj):
    second_neigh_list = []
    half_subgraph_size = len(subgraph) + 1

    for i in range(half_subgraph_size):
        if i == half_subgraph_size - 1:
            cur_neigh = subgraph[i-1]
        else:
            cur_neigh = subgraph[i]
        cur_neigh_first_adj = np.where(adj[cur_neigh] == 1)
        cur_neigh_first_adj = cur_neigh_first_adj[0].tolist()

        if len(cur_neigh_first_adj) <= 1:
            second_neigh = cur_neigh
        else:
            second_neigh = random.choice(cur_neigh_first_adj)
            retry_time = 0
            while ((second_neigh == target_node) or (second_neigh in subgraph)) and retry_time < 10:
                second_neigh = random.choice(cur_neigh_first_adj)
                retry_time += 1
        second_neigh_list.append(second_neigh)

    return second_neigh_list




def get_second_adj_5(dgl_graph, adj, subgraph_size):
    """Generate the second view's subgraph with the 1/2 first-order and 1/2 second-order neighbor connected to 1-st. """
    all_idx = list(range(dgl_graph.number_of_nodes()))
    subgraphs = []
    adj_2 = adj.dot(adj)
    adj = np.array(adj.todense())
    adj_2 = np.array(adj_2.todense())
    row, col = np.diag_indices_from(adj_2)
    zeros = np.zeros(adj_2.shape[0])
    adj_2[row, col] = np.array(zeros)
    adj = adj.squeeze()
    adj_2 = adj_2.squeeze()
    for node_id in all_idx:
        first_adj = np.where(adj[node_id] == 1)
        second_adj = np.where(adj_2[node_id] != 0)
        first_adj = first_adj[0].tolist()
        second_adj = second_adj[0].tolist()
        if len(first_adj) < ((subgraph_size - 1) // 2):
            subgraphs.append(list(np.random.choice(first_adj, (subgraph_size - 1) // 2, replace=True)))
            if len(second_adj) == 0:
                first_adj.append(node_id)
                subgraphs[node_id].extend(list(np.random.choice(first_adj, (subgraph_size) // 2, replace=True)))
            elif len(second_adj) < (subgraph_size - 1) // 2:
                subgraphs[node_id].extend(list(np.random.choice(second_adj, (subgraph_size) // 2, replace=True)))
            else:
                second_neigh_list = find_1to2_neigh(node_id, subgraphs[node_id], adj)
                subgraphs[node_id].extend(list(second_neigh_list))
        else:
            if len(second_adj) == 0:
                first_adj.append(node_id)
                if len(first_adj) < subgraph_size - 1:
                    subgraphs.append(list(np.random.choice(first_adj, (subgraph_size - 1), replace=True)))
                else:
                    subgraphs.append(list(np.random.choice(first_adj, (subgraph_size - 1), replace=False)))
            elif len(second_adj) < (subgraph_size) // 2 :
                subgraphs.append(list(np.random.choice(first_adj, (subgraph_size - 1) // 2, replace=False)))
                subgraphs[node_id].extend(list(np.random.choice(second_adj, (subgraph_size) // 2, replace=True)))
            else:
                subgraphs.append(list(np.random.choice(first_adj, (subgraph_size - 1) // 2, replace=False)))
                second_neigh_list = find_1to2_neigh(node_id, subgraphs[node_id], adj)
                subgraphs[node_id].extend(list(second_neigh_list))


        subgraphs[node_id].append(node_id)
    return subgraphs



def get_second_adj_6(dgl_graph, adj, subgraph_size_1, subgraph_size_2):
    """Generate the second view's subgraph with the 1/2 first-order and 1/2 second-order neighbor connected to 1-st. """
    """Return two view subgraph simultaneously"""
    all_idx = list(range(dgl_graph.number_of_nodes()))
    adj = np.array(adj.todense())
    row, col = np.diag_indices_from(adj)
    zeros = np.zeros(adj.shape[0])
    adj[row, col] = np.array(zeros)
    adj = adj.squeeze()
    subgraphs_1 = []
    subgraphs_2 = []

    for target_node in all_idx:
        first_adj = np.where(adj[target_node] == 1)
        first_adj = list(first_adj[0])
        if len(first_adj) == 0:  # 孤立点
            first_adj.append(target_node)
            subgraphs_1.append(list(np.random.choice(first_adj, subgraph_size_1, replace=True)))
            subgraphs_2.append(list(np.random.choice(first_adj, subgraph_size_2, replace=True)))

        else:
            cur_subgraph_1 = []
            cur_subgraph_2 = []
            if len(first_adj) < subgraph_size_1 - 1:
                cur_subgraph_1.extend(first_adj)
                cur_subgraph_1.extend(list(np.random.choice(first_adj, subgraph_size_1 - len(first_adj) - 1, replace=True)))
            else:
                cur_subgraph_1.extend(list(np.random.choice(first_adj, subgraph_size_1 - 1, replace=False)))


            subgraphs_1_rest = list(OrderedDict.fromkeys(cur_subgraph_1.copy())) # 去除重复元素
            subgraphs_1_rest = [i for i in subgraphs_1_rest if i != target_node] # 去除当前点

            second_adj_list = []
            for ego_node in subgraphs_1_rest:
                second_adj = np.where(adj[ego_node] == 1)[0]
                second_adj_list.extend(second_adj)
            # second_adj_list = sum(second_adj_list, [])

            second_adj_rest = list(OrderedDict.fromkeys(second_adj_list))  # 去除重复元素
            second_adj_rest = [i for i in second_adj_rest if i != target_node]  # 去除当前点

            # temp = cur_subgraph_1.copy()
            # subgraphs_2.append(temp)
            cur_subgraph_2.extend(cur_subgraph_1)

            if len(second_adj_rest) == 0:
                first_adj.append(target_node)
                cur_subgraph_2.extend(list(np.random.choice(first_adj, subgraph_size_2 - subgraph_size_1, replace=True)))
            elif len(second_adj_rest) < (subgraph_size_2 - subgraph_size_1):
                cur_subgraph_2.extend(second_adj_rest)
                # second_adj_rest_add = second_adj_rest.copy()
                # second_adj_rest_add.append(target_node)
                cur_subgraph_2.extend(list(np.random.choice(second_adj_rest, subgraph_size_2 - subgraph_size_1 - len(second_adj_rest), replace=True)))
            else:
                cur_subgraph_2.extend(list(np.random.choice(second_adj_rest, (subgraph_size_2 - subgraph_size_1), replace=False)))

            cur_subgraph_1.append(target_node)
            cur_subgraph_2.append(target_node)
            subgraphs_1.append(cur_subgraph_1)
            subgraphs_2.append(cur_subgraph_2)

    return subgraphs_1, subgraphs_2



def get_second_adj_7(dgl_graph, adj, subgraph_size):
    """Generate the second view's subgraph with the 1/2 first-order and 1/2 second-order neighbor connected to 1-st. """
    all_idx = list(range(dgl_graph.number_of_nodes()))
    adj = np.array(adj.todense())
    row, col = np.diag_indices_from(adj)
    zeros = np.zeros(adj.shape[0])
    adj[row, col] = np.array(zeros)
    adj = adj.squeeze()

    subgraphs_2 = []
    subgraph_size_1 = int(subgraph_size // 2)
    subgraph_size_2 = subgraph_size

    for target_node in all_idx:
        first_adj = np.where(adj[target_node] == 1)
        first_adj = list(first_adj[0])
        if len(first_adj) == 0:  # 孤立点
            first_adj.append(target_node)
            subgraphs_2.append(list(np.random.choice(first_adj, subgraph_size_2, replace=True)))

        else:
            cur_subgraph_1 = []
            cur_subgraph_2 = []
            if len(first_adj) < subgraph_size_1 - 1:
                cur_subgraph_1.extend(first_adj)
                cur_subgraph_1.extend(list(np.random.choice(first_adj, subgraph_size_1 - len(first_adj) - 1, replace=True)))
            else:
                cur_subgraph_1.extend(list(np.random.choice(first_adj, subgraph_size_1 - 1, replace=False)))


            subgraphs_1_rest = list(OrderedDict.fromkeys(cur_subgraph_1.copy())) # 去除重复元素
            subgraphs_1_rest = [i for i in subgraphs_1_rest if i != target_node] # 去除当前点

            second_adj_list = []
            for ego_node in subgraphs_1_rest:
                second_adj = np.where(adj[ego_node] == 1)[0]
                second_adj_list.extend(second_adj)
            # second_adj_list = sum(second_adj_list, [])

            second_adj_rest = list(OrderedDict.fromkeys(second_adj_list))  # 去除重复元素
            second_adj_rest = [i for i in second_adj_rest if i != target_node]  # 去除当前点

            # temp = cur_subgraph_1.copy()
            # subgraphs_2.append(temp)
            cur_subgraph_2.extend(cur_subgraph_1)

            if len(second_adj_rest) == 0:
                first_adj.append(target_node)
                cur_subgraph_2.extend(list(np.random.choice(first_adj, subgraph_size_2 - subgraph_size_1, replace=True)))
            elif len(second_adj_rest) < (subgraph_size_2 - subgraph_size_1):
                cur_subgraph_2.extend(second_adj_rest)
                cur_subgraph_2.extend(list(np.random.choice(second_adj_rest, subgraph_size_2 - subgraph_size_1 - len(second_adj_rest), replace=True)))
            else:
                cur_subgraph_2.extend(list(np.random.choice(second_adj_rest, (subgraph_size_2 - subgraph_size_1), replace=False)))

            cur_subgraph_2.append(target_node)
            subgraphs_2.append(cur_subgraph_2)


    return subgraphs_2




def get_third_adj(dgl_graph, adj, subgraph_size):
    """Generate the second view's subgraph with the first-ordtart_prob_2', type=float, help='RWR restart probability on view 2', er and second-order neighbor."""
    all_idx = list(range(dgl_graph.number_of_nodes()))
    subgraphs = []
    adj_2 = adj.dot(adj)
    adj_3 = adj_2.dot(adj)
    adj = np.array(adj.todense())
    adj_2 = np.array(adj_2.todense())
    adj_3 = np.array(adj_3.todense())
    row_2, col_2 = np.diag_indices_from(adj_2)
    zeros = np.zeros(adj_2.shape[0])
    adj_2[row_2, col_2] = np.array(zeros)
    row_3, col_3 = np.diag_indices_from(adj_3)
    zeros = np.zeros(adj_3.shape[0])
    adj_3[row_3, col_3] = np.array(zeros)
    adj = adj.squeeze()
    adj_2 = adj_2.squeeze()
    adj_3 = adj_3.squeeze()
    for node_id in all_idx:
        first_adj = np.where(adj[node_id] == 1)
        second_adj = np.where(adj_2[node_id] != 0)
        third_adj = np.where(adj_3[node_id] != 0)
        first_adj = first_adj[0].tolist()
        second_adj = second_adj[0].tolist()
        third_adj = third_adj[0].tolist()
        if len(first_adj) > 0:
            subgraphs.append(list(np.random.choice(first_adj, 2, replace=True)))
            if len(second_adj) > 0:
                if len(second_adj) < 2:
                    subgraphs[node_id].extend(list(np.random.choice(second_adj, 2, replace=True)))
                    if len(third_adj) == 0:
                        subgraphs[node_id].extend(list(np.random.choice(first_adj, 1, replace=True)))
                        subgraphs[node_id].extend(list(np.random.choice(second_adj, 2, replace=True)))
                    elif 0 < len(third_adj) < 3:
                        subgraphs[node_id].extend(
                            list(np.random.choice(third_adj, 3, replace=True)))
                    elif len(third_adj) >= 3:
                        subgraphs[node_id].extend(list(np.random.choice(third_adj, 3, replace=False)))
                else:
                    subgraphs[node_id].extend(
                        list(np.random.choice(second_adj, 2, replace=False)))
                    if len(third_adj) == 0:
                        subgraphs[node_id].extend(list(np.random.choice(first_adj, 1, replace=True)))
                        subgraphs[node_id].extend(list(np.random.choice(second_adj, 2, replace=True)))
                    elif 0 < len(third_adj) < 3:
                        subgraphs[node_id].extend(
                            list(np.random.choice(third_adj, 3, replace=True)))
                    elif len(third_adj) >= 3:
                        subgraphs[node_id].extend(list(np.random.choice(third_adj, 3, replace=False)))
            else:
                subgraphs[node_id].extend(
                    list(np.random.choice(first_adj + [node_id], 5, replace=True)))
        else:
            first_adj.append(node_id)
            subgraphs.append(list(np.random.choice(first_adj, subgraph_size - 1, replace=True)))
        subgraphs[node_id].append(node_id)
    return subgraphs



def get_third_adj_2(dgl_graph, adj, subgraph_size):
    """Generate the subgraph with the 1/3 first-order, 1/3 second-order and 1/3 third-order neighbor ."""
    # subgraph_size=4 / 7 / 10 / ...
    if subgraph_size % 3 != 1:
        raise NotImplementedError
    all_idx = list(range(dgl_graph.number_of_nodes()))
    subgraphs = []
    adj_2 = adj.dot(adj)
    adj_3 = adj_2.dot(adj)
    adj = np.array(adj.todense())
    adj_2 = np.array(adj_2.todense())
    adj_3 = np.array(adj_3.todense())
    row_2, col_2 = np.diag_indices_from(adj_2)
    zeros = np.zeros(adj_2.shape[0])
    adj_2[row_2, col_2] = np.array(zeros)
    row_3, col_3 = np.diag_indices_from(adj_3)
    zeros = np.zeros(adj_3.shape[0])
    adj_3[row_3, col_3] = np.array(zeros)
    adj = adj.squeeze()
    adj_2 = adj_2.squeeze()
    adj_3 = adj_3.squeeze()

    for node_id in all_idx:
        first_adj = np.where(adj[node_id] == 1)
        second_adj = np.where(adj_2[node_id] != 0)
        third_adj = np.where(adj_3[node_id] != 0)
        first_adj = first_adj[0].tolist()
        second_adj = second_adj[0].tolist()
        third_adj = third_adj[0].tolist()
        if len(first_adj) < subgraph_size // 3:
            subgraphs.append(list(np.random.choice(first_adj, subgraph_size // 3, replace=True)))
            if len(second_adj) == 0:
                first_adj.append(node_id)
                subgraphs[node_id].extend(list(np.random.choice(first_adj, subgraph_size // 3, replace=True)))
            elif len(second_adj) < subgraph_size // 3:
                subgraphs[node_id].extend(list(np.random.choice(second_adj, subgraph_size // 3, replace=True)))
            else:
                subgraphs[node_id].extend(list(np.random.choice(second_adj, subgraph_size // 3, replace=False)))
            if len(third_adj) == 0:
                subgraphs[node_id].extend(list(np.random.choice(first_adj, subgraph_size // 3, replace=True)))
            elif len(third_adj) < subgraph_size // 3:
                subgraphs[node_id].extend(list(np.random.choice(third_adj, subgraph_size // 3, replace=True)))
            else:
                subgraphs[node_id].extend(list(np.random.choice(third_adj, subgraph_size // 3, replace=False)))
        else:
            if len(second_adj) == 0:
                # first_adj.append(node_id)
                if len(first_adj) < subgraph_size - 1:
                    subgraphs.append(list(np.random.choice(first_adj, (subgraph_size - 1), replace=True)))
                else:
                    subgraphs.append(list(np.random.choice(first_adj, (subgraph_size - 1), replace=False)))
            else:
                subgraphs.append(list(np.random.choice(first_adj, subgraph_size // 3, replace=False)))
                if len(second_adj) < subgraph_size // 3:
                    subgraphs[node_id].extend(list(np.random.choice(second_adj, subgraph_size // 3, replace=True)))
                else:
                    subgraphs[node_id].extend(list(np.random.choice(second_adj, subgraph_size // 3, replace=False)))
                if len(third_adj) == 0:
                    subgraphs[node_id].extend(list(np.random.choice(second_adj, subgraph_size // 3, replace=True)))
                elif len(third_adj) < subgraph_size // 3:
                    subgraphs[node_id].extend(list(np.random.choice(third_adj, subgraph_size // 3, replace=True)))
                else:
                    subgraphs[node_id].extend(list(np.random.choice(third_adj, subgraph_size // 3, replace=False)))

        subgraphs[node_id].append(node_id)
    return subgraphs




def generate_rwr_subgraph(dgl_graph, subgraph_size, restart_prob):
    """Generate subgraph with RWR algorithm."""
    all_idx = list(range(dgl_graph.number_of_nodes()))
    reduced_size = subgraph_size - 1
    traces = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, all_idx, restart_prob=restart_prob, max_nodes_per_seed=subgraph_size*3)
    subv = []

    for i,trace in enumerate(traces):
        subv.append(torch.unique(torch.cat(trace),sorted=False).tolist())
        retry_time = 0
        while len(subv[i]) < reduced_size:
            cur_trace = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, [i], restart_prob=(restart_prob/2), max_nodes_per_seed=subgraph_size*5)
            subv[i] = torch.unique(torch.cat(cur_trace[0]),sorted=False).tolist()
            retry_time += 1
            if retry_time >10:
                subv[i] = (subv[i] * reduced_size)
            # if (len(subv[i]) <= 2) and (retry_time >10):
            #     subv[i] = (subv[i] * reduced_size)
        subv[i] = subv[i][:reduced_size]
        subv[i].append(i)

    return subv



def generate_subgraph(args, dgl_graph, A, subgraph_size_1, subgraph_size_2, cnt):
    """Generate subgraph with RWR/first & second & third-neiborhood algorithm."""
    # subgraph_size_1 = args.subgraph_size_1
    # subgraph_size_2 = args.subgraph_size_2
    restart_prob_1 = args.restart_prob_1
    restart_prob_2 = args.restart_prob_2

    # np.random.seed(args.seed+cnt)

    if args.subgraph_mode == 'random':
        subgraphs_1 = generate_rwr_subgraph(dgl_graph, subgraph_size_1, restart_prob=restart_prob_1)
        subgraphs_2 = generate_rwr_subgraph(dgl_graph, subgraph_size_2, restart_prob=restart_prob_2)
    elif args.subgraph_mode == '1+1':
        subgraphs_1 = get_first_adj(dgl_graph, A, subgraph_size_1)
        subgraphs_2 = get_first_adj(dgl_graph, A, subgraph_size_2)
    elif args.subgraph_mode == '1+2':
        subgraphs_1 = get_first_adj(dgl_graph, A, subgraph_size_1)
        subgraphs_2 = get_second_adj_1(dgl_graph, A, subgraph_size_2)
    elif args.subgraph_mode == '2':
        subgraphs_1 = get_first_adj(dgl_graph, A, subgraph_size_1)
        subgraphs_2 = get_second_adj_2(dgl_graph, A, subgraph_size_2)
    elif args.subgraph_mode == '1/2+1/2':
        subgraphs_1 = get_first_adj(dgl_graph, A, subgraph_size_1)
        subgraphs_2 = get_second_adj_3(dgl_graph, A, subgraph_size_2)
    elif args.subgraph_mode == '1+new1/2':
        subgraphs_1 = get_first_adj(dgl_graph, A, subgraph_size_1)
        subgraphs_2 = get_second_adj_5(dgl_graph, A, subgraph_size_2)
    elif args.subgraph_mode == '1+2+3':
        subgraphs_1 = get_first_adj(dgl_graph, A, subgraph_size_1)
        subgraphs_2 = get_third_adj(dgl_graph, A, subgraph_size_2)
    elif args.subgraph_mode == '1/3+1/3+1/3':
        subgraphs_1 = get_first_adj(dgl_graph, A, subgraph_size_1)
        subgraphs_2 = get_third_adj_2(dgl_graph, A, subgraph_size_2)
    elif args.subgraph_mode == 'same':
        subgraphs_1 = get_second_adj_3(dgl_graph, A, subgraph_size_1)
        subgraphs_2 = get_second_adj_3(dgl_graph, A, subgraph_size_2)
    elif args.subgraph_mode == '1+only2':
        subgraphs_1 = get_first_adj(dgl_graph, A, subgraph_size_1)
        subgraphs_2 = get_second_adj_4(dgl_graph, A, subgraph_size_2)
    elif args.subgraph_mode == 'only2+1':
        subgraphs_1 = get_second_adj_4(dgl_graph, A, subgraph_size_1)
        subgraphs_2 = get_first_adj(dgl_graph, A, subgraph_size_2)
    # elif args.subgraph_mode == '1_small':
    #     subgraphs_1 = get_first_adj_small(dgl_graph, A, subgraph_size_1, degree, avg_degree)
    #     subgraphs_2 = get_second_adj_3(dgl_graph, A, subgraph_size_2)
    # elif args.subgraph_mode == '1_high':
    #     subgraphs_1 = get_first_adj_high(dgl_graph, A, subgraph_size_1, degree, avg_degree)
    #     subgraphs_2 = get_second_adj_3(dgl_graph, A, subgraph_size_2)
    elif args.subgraph_mode == '2+2':
        subgraphs_1 = get_second_adj_7(dgl_graph, A, subgraph_size_1)
        subgraphs_2 = get_second_adj_7(dgl_graph, A, subgraph_size_2)
    elif args.subgraph_mode == '1+near':
        subgraphs_1, subgraphs_2 = get_second_adj_6(dgl_graph, A, subgraph_size_1, subgraph_size_2)
    elif args.subgraph_mode == '1+1connect2':
        subgraphs_1 = get_first_adj(dgl_graph, A, subgraph_size_1)
        subgraphs_2 = get_second_adj_7(dgl_graph, A, subgraph_size_2)
    else:
        raise NotImplementedError


    return subgraphs_1, subgraphs_2




