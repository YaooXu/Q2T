import pickle

import os
import random
from collections import defaultdict
from typing import List

import numpy as np
import torch
from torch_geometric.data import Data

from utils.util import Integer, tuple2list, list2tuple, TYPE_TO_IDX, OFFSET, ENT_CLS, REL_CLS, TGT, VAR

query_structure_to_type = {('e', ('r',)): '1p',
                           ('e', ('r', 'r')): '2p',
                           ('e', ('r', 'r', 'r')): '3p',
                           (('e', ('r',)), ('e', ('r',))): '2i',
                           (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                           ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                           (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                           (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                           (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                           ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                           (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                           (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                           (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                           ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                           # ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                           # ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
                           }
query_type_to_structure = {value: key for key, value in query_structure_to_type.items()}
all_tasks = list(
    query_type_to_structure.keys())

query_type_to_struct_feat = {
    '1p': [[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 1]],
    '2p': [[0, 0, 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 1]],
    '3p': [[0, 1, 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 1]],
    '2i': [[0, 0, 0, 0],
           [0, 0, 0, 1],
           [0, 0, 0, 1]],
    '3i': [[0, 0, 0, 1],
           [0, 0, 0, 1],
           [0, 0, 0, 1]],
    '2in': [[0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, -1]],
    '3in': [[0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, -1]],
    'inp': [[0, 0, 1, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]],
    'pni': [[0, 1, 0, 0],
            [0, 0, 0, -1],
            [0, 0, 0, 1]],
    'pin': [[0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]],
    'ip': [[0, 0, 1, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 1]],
    'pi': [[0, 1, 0, 0],
           [0, 0, 0, 1],
           [0, 0, 1, 0]],
}
query_type_to_struct_feat['2u-DNF'] = query_type_to_struct_feat['1p']
query_type_to_struct_feat['up-DNF'] = query_type_to_struct_feat['2p']


class QueryNode:
    def __init__(self, children=None, ent_idx=None, rels=None, union=None):
        self.children: List[QueryNode] = children
        self.ent_idx = ent_idx
        self.rels = rels
        self.union = union

    def is_anchor_node(self):
        return self.ent_idx is not None

    def is_rels_node(self):
        return self.rels is not None

    @staticmethod
    def is_union(tup):
        # ('u',)
        return tup == (-1,)

    @staticmethod
    def is_rel_chain(tup):
        for item in tup:
            if type(item) is tuple or item == -1:
                return False
        return True

    def is_rels_node_in_children(self):
        for child in self.children:
            if child.is_rels_node():
                return True
        return False

    @staticmethod
    def construct_from_query(query):
        if type(query) is int:
            # anchor node
            ent_idx = query
            return QueryNode(ent_idx=ent_idx)
        elif QueryNode.is_union(query):
            return QueryNode(union=True)
        elif QueryNode.is_rel_chain(query):
            # rel path
            rels = query
            return QueryNode(rels=rels)
        else:
            children = []
            queryNode = QueryNode(children=children)

            for item in query:
                children.append(QueryNode.construct_from_query(item))

            return queryNode


class DagNode:
    def __init__(self, num_nodes: Integer, children=None, ent_idx=None):
        self.idx = num_nodes.val
        num_nodes += 1

        self.children: List[(int, DagNode)] = [] if children is None else children
        self.ent_idx = ent_idx
        self.operation = None

    def __repr__(self):
        return str(self.idx)

    def set_parent(self, edges,
                   parent_node: 'DagNode', rel):
        # add this node to parent node children list
        edges.append((self.idx, rel, parent_node.idx))
        parent_node.children.append((rel, self))

    @staticmethod
    def construct_from_chain(edges,
                             num_nodes: Integer, end_node, rels, parent_node: 'DagNode'):
        # given end_node, (r1, r2), parent_node
        # construct node chain like end_node <- v1 <- parent_node

        dagNodes = list(reversed([DagNode(num_nodes=num_nodes) for _ in range(len(rels) - 1)]))

        dagNodes = [end_node] + dagNodes

        p = parent_node
        for dagNode, rel in zip(reversed(dagNodes), reversed(rels)):
            dagNode.set_parent(edges,
                               p, rel)

            p = dagNode

        return parent_node

    @staticmethod
    def __construct_from_query(edges, node_to_ent, node_to_operation,
                               queryNode: QueryNode, num_nodes=Integer(0), parent_node=None):
        if queryNode.is_anchor_node():
            cur_node = parent_node
            cur_node.ent_idx = queryNode.ent_idx

            node_to_ent[cur_node.idx] = cur_node.ent_idx

            return cur_node
        else:
            stack: List[DagNode] = []

            if queryNode.is_rels_node_in_children():
                # (...), (r, r, r)
                # have to construct a new node to pass to subquery
                sub_dag_root = DagNode(num_nodes)
            else:
                # don't need to create new node
                sub_dag_root = parent_node

            for i, child_queryNode in enumerate(queryNode.children):
                if i == 1 and child_queryNode.is_rels_node():
                    rels = child_queryNode.rels
                    end_node = stack.pop()
                    DagNode.construct_from_chain(edges,
                                                 num_nodes, end_node, rels, parent_node)
                elif i == len(queryNode.children) - 1 and child_queryNode.union is True:
                    parent_node.operation = 1
                    node_to_operation[parent_node.idx] = 1
                else:
                    stack.append(DagNode.__construct_from_query(edges, node_to_ent, node_to_operation,
                                                                child_queryNode, num_nodes, sub_dag_root))

            if len(parent_node.children) > 1 and parent_node.operation is None:
                parent_node.operation = 0
                node_to_operation[parent_node.idx] = 0

        return parent_node

    @staticmethod
    def construct_from_query(query):
        query_root = QueryNode.construct_from_query(query=query)

        num_nodes = Integer(0)
        root = DagNode(num_nodes)

        edges = []
        node_to_ent = {}
        node_to_operation = defaultdict(int)
        operation_to_node = defaultdict(list)

        DagNode.__construct_from_query(edges, node_to_ent, node_to_operation,
                                       query_root, num_nodes, root)

        for node, operation in node_to_operation.items():
            operation_to_node[operation].append(node)

        # print(edges)
        # print(node_to_ent)
        # print(operation_to_node)
        return num_nodes.val, edges, node_to_ent, operation_to_node


def get_inv_rel(rel):
    return rel ^ 1


def topoSort(edges) -> (dict, dict):
    if type(edges) is np.ndarray:
        edges = edges.tolist()

    inv_G = defaultdict(list)
    in_degrees = defaultdict(int)

    src_node_to_edge_idxes = {}

    for i, edge in enumerate(edges):
        s, r, t = edge
        inv_G[t].append(s)
        src_node_to_edge_idxes[s] = i

        if s not in in_degrees:
            in_degrees[s] = 0
        if t not in in_degrees:
            in_degrees[t] = 0

        in_degrees[t] += 1

    # get node layer by BFS from target node
    q = [0]

    # layer flag
    FLAG = -1
    q.append(FLAG)

    cur_layer = 0
    layer_to_nodes = defaultdict(list)
    layer_to_edge_idxes = defaultdict(list)

    while q:
        u = q.pop(0)

        if u == FLAG:
            if len(q) == 0:
                break

            cur_layer += 1
            q.append(FLAG)
            continue

        layer_to_nodes[cur_layer].append(u)
        if u in src_node_to_edge_idxes:
            # nodes in last layer don't have edges
            layer_to_edge_idxes[cur_layer].append(src_node_to_edge_idxes[u])

        for v in inv_G[u]:
            q.append(v)

    max_layer_idx = max(layer_to_nodes.keys())

    layer_to_edge_idxes = {max_layer_idx - k: v for k, v in layer_to_edge_idxes.items()}
    layer_to_nodes = {max_layer_idx - k: v for k, v in layer_to_nodes.items()}

    # # for convenience, layer_to_nodes don't contain target nodes
    # layer_to_nodes.pop(cur_layer)

    return layer_to_nodes, layer_to_edge_idxes


TYPE_TO_DIST_MAT = {}


def floyd(adj_mat: np.ndarray, layers, seq_num_nodes, node_types, query_type, enc_dist):
    if query_type not in TYPE_TO_DIST_MAT:
        num_nodes = len(adj_mat)

        dist = adj_mat.copy()
        dist[dist == 0] = 1000
        dist[range(num_nodes), range(num_nodes)] = 0

        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])

        dist[0, 1] = dist[1, 0] = 7

        # 8, 9
        dist[0, 2:seq_num_nodes] = dist[2:seq_num_nodes, 0] = 8 + node_types[2:seq_num_nodes]

        # 10, 11
        dist[1, 2:seq_num_nodes] = dist[2:seq_num_nodes, 1] = 10 + node_types[2:seq_num_nodes]

        dist[dist == 1000] = 0

        if enc_dist == 'd':
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if layers[i] < layers[j]:
                        dist[i, j] = -dist[i, j]
            dist[0, 1] = -dist[0, 1]
        TYPE_TO_DIST_MAT[query_type] = dist.astype(int)

    return TYPE_TO_DIST_MAT[query_type]


def convert_query_to_pyg(query, bi_dir=False):
    num_nodes, edges, node_to_ent, operation_to_node = DagNode.construct_from_query(query)
    edges = np.array(edges)
    src, rel, dst = edges.T

    # create bidirectional graph
    if bi_dir:
        src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
        rel = np.concatenate((rel, get_inv_rel(rel)))

    edge_index = np.stack((src, dst))
    edge_type = rel

    target_to_in_edges = defaultdict(list)
    for r, t in zip(edge_type, edge_index[1]):
        target_to_in_edges[t].append(r)

    tail_to_edge_idxes = defaultdict(list)
    for i, t in enumerate(dst):
        tail_to_edge_idxes[t].append(i)

    # edge_index is array now, it will be converted to tensor in getitem
    data = Data(edge_index=edge_index)
    data.num_nodes = num_nodes
    data.num_edges_ = len(edge_type)
    data.edge_type = edge_type
    data.node_to_ent = node_to_ent
    data.operation_to_node = operation_to_node
    data.target_to_in_edges = target_to_in_edges
    # data.tail_to_edge_idxes = tail_to_edge_idxes
    data.query_idx_to_union_nodes = {}

    data.origin_query = query

    anchor_node_idxes = set(node_to_ent.keys())
    target_node_idx = {0}
    inter_node_idxes = list(set(range(num_nodes)) - anchor_node_idxes - target_node_idx)
    # used in pseudo label predicting
    data.inter_node_to_ent = {idx: -1 for idx in inter_node_idxes}
    data.target_node_idxes = 0

    if not bi_dir:
        # topology structure
        layer_to_nodes, layer_to_edge_idxes = topoSort(edges)
    else:
        layer_to_nodes, layer_to_edge_idxes = {}, {}

    data.layer_to_nodes = layer_to_nodes
    data.layer_to_edge_idxes = layer_to_edge_idxes

    node_to_layer = {}
    for layer, nodes in layer_to_nodes.items():
        node_to_layer.update({node: layer for node in nodes})

    # e.g. {1: {0: [0, 1, 2]}})
    layer_to_tail_to_edge_idxes = defaultdict(dict)
    for tail, edge_idxes in tail_to_edge_idxes.items():
        layer_to_tail_to_edge_idxes[node_to_layer[tail]][tail] = edge_idxes

    data.layer_to_tail_to_edge_idxes = layer_to_tail_to_edge_idxes

    return data


def __handle_neg_query(query):
    is_rel_path = True
    for sub_query in query:
        if type(sub_query) is list:
            is_rel_path = False
            __handle_neg_query(sub_query)

    if is_rel_path:
        # e.g. (171,), (87, -2)
        if query[-1] == -2:
            # is neg
            # (87, -2) -> (-87,)
            query[-2] = -query[-2]
            query.pop(-1)


def encode_query(query, query_type, enc_dist='d'):
    if query_type in ['2u-DNF', 'up-DNF']:
        if query_type == '2u-DNF':
            queries = [query[0], query[1]]
            query_type = '1p'
        elif query_type == 'up-DNF':
            query = tuple2list(query)
            query, rel = query[0], query[1][0]
            query[0][-1].append(rel)
            query[1][-1].append(rel)
            query = list2tuple(query)
            queries = [query[0], query[1]]

        datas = []
        for query in queries:
            data = __encode_query(query, query_type, enc_dist)
            datas.append(data)

        # case other information are all the same
        data = datas[0]
        data['x'] = [datas[0]['x'], datas[1]['x']]
        return data
    elif 'n' in query_type:
        query = tuple2list(query)
        __handle_neg_query(query)
        query = list2tuple(query)
        # print(query)
        return __encode_query(query, query_type, enc_dist)
    else:
        return __encode_query(query, query_type, enc_dist)


def __encode_query(query, query_type, enc_dist):
    num_nodes, edges, node_to_ent, operation_to_node = DagNode.construct_from_query(query)
    layer_to_nodes, layer_to_edge_idxes = topoSort(edges)

    idx_convert = {}

    num_ent_nodes = num_nodes
    num_rel_nodes = len(edges)

    MAX_LEN = 10

    # ENT_CLS, REL_CLS
    seq_num_nodes = num_ent_nodes + num_rel_nodes + 2
    attention_mask = np.array([1] * seq_num_nodes + [0] * (MAX_LEN - seq_num_nodes))

    # x: [ENT_CLS, REL_CLS, ent1, ent2, ..., tgt_ent , rel1, rel2, ..., PAD, ...,]
    x = np.zeros(MAX_LEN, dtype=int)
    ENT_CLS_idx, REL_CLS_idx = 0, 1
    tgt_node_idx = 2 + num_ent_nodes - 1

    idx_convert[0] = tgt_node_idx

    # PAD = 0
    # VAR = 1
    # TGT = 2
    # ENT_CLS = 3
    # REL_CLS = 4
    # OFFSET = 5
    # [4, ...] denotes ent, [..., -1] denotes edge
    x[ENT_CLS_idx] = ENT_CLS
    x[REL_CLS_idx] = REL_CLS

    targets = np.zeros(MAX_LEN, dtype=int)
    targets[tgt_node_idx] = 1

    # 0: ent, 1: rel
    node_types = np.zeros(MAX_LEN, dtype=int)
    node_types[ENT_CLS_idx] = 0
    node_types[REL_CLS_idx] = 1

    layers = np.zeros(MAX_LEN, dtype=int)
    layers[ENT_CLS_idx] = 0
    layers[REL_CLS_idx] = 0

    negs = np.zeros(MAX_LEN, dtype=int)

    cur_edge_idx = 2 + num_ent_nodes
    cur_node_idx = 2

    adj_mat = np.zeros([MAX_LEN, MAX_LEN], dtype=int)
    tgt_type = {
        "1p": 0,
        "2p": 0,
        "3p": 0,
        'inp': 0,
        'ip': 0,
        '2u-DNF': 0,
        'up-DNF': 0,
        "2i": 1,
        "3i": 1,
        'pi': 1,
        '2in': 2,
        'pni': 2,
        'pin': 2,
        '3in': 2,
    }[query_type]

    in_degs = np.zeros(MAX_LEN, dtype=int)
    out_degs = np.zeros(MAX_LEN, dtype=int)

    for layer in sorted(layer_to_edge_idxes.keys()):
        for edge_idx in layer_to_edge_idxes[layer]:
            edge = edges[edge_idx]
            h, r, t = edge

            # get idx in seq
            for item in [h, t]:
                if item not in idx_convert:
                    idx_convert[item] = cur_node_idx
                    cur_node_idx += 1

            h_idx, t_idx = idx_convert[h], idx_convert[t]
            rel_idx = cur_edge_idx
            cur_edge_idx += 1

            if h in node_to_ent:
                # ent (anchor node)
                ent = node_to_ent[h]

                x[h_idx] = ent + OFFSET
                node_types[h_idx] = 0
            else:
                x[h_idx] = VAR

            if t_idx == tgt_node_idx:
                x[t_idx] = TGT
            else:
                x[t_idx] = VAR

            if r < 0:
                # neg r
                tgt_type = 1
                r = -r
                # x[rel_idx] = NEG
                x[rel_idx] = -(r + 1)

                negs[rel_idx] = 1
                negs[h_idx] = 2
            else:
                x[rel_idx] = -(r + 1)

            node_types[rel_idx] = 1

            out_degs[h_idx] += 1
            in_degs[rel_idx] += 1
            out_degs[rel_idx] += 1
            in_degs[t_idx] += 1

            adj_mat[h_idx, rel_idx] = adj_mat[rel_idx, h_idx] = 1
            adj_mat[t_idx, rel_idx] = adj_mat[rel_idx, t_idx] = 1

            layers[h_idx] = 1 + layer * 2
            layers[rel_idx] = 2 + layer * 2
            layers[t_idx] = 3 + layer * 2

    operators = np.zeros(MAX_LEN, dtype=int)
    for idx in operation_to_node[0]:
        operators[idx_convert[idx]] = 1

    in_degs[ENT_CLS_idx] = out_degs[ENT_CLS_idx] = num_ent_nodes + num_rel_nodes
    in_degs[REL_CLS_idx] = out_degs[REL_CLS_idx] = num_ent_nodes + num_rel_nodes

    # for debugging, permutation according to layers
    idxes = np.argsort(layers[:seq_num_nodes])
    x[:seq_num_nodes] = x[idxes]
    layers[:seq_num_nodes] = layers[idxes]
    node_types[:seq_num_nodes] = node_types[idxes]
    negs[:seq_num_nodes] = negs[idxes]
    operators[:seq_num_nodes] = operators[idxes]
    in_degs[:seq_num_nodes] = in_degs[idxes]
    out_degs[:seq_num_nodes] = out_degs[idxes]

    adj_mat[:seq_num_nodes] = adj_mat[idxes]
    adj_mat[:, :seq_num_nodes] = adj_mat[:, idxes]

    data = {
        'x': x.tolist(),
        'query_type': [TYPE_TO_IDX.get(query_type, -1)],
        'node_types': node_types.tolist(),
        'tgt_type': [tgt_type],
        'operators': operators.tolist(),
        'layers': layers.tolist(),
        'targets': targets.tolist(),
        'in_degs': in_degs.tolist(),
        'out_degs': out_degs.tolist(),
        'negs': negs.tolist(),
    }

    if enc_dist in ['u', 'd', 'no']:
        data.update({
            'attention_mask': attention_mask.tolist(),
            'dist_mat': floyd(adj_mat, layers, seq_num_nodes, node_types, query_type, enc_dist).tolist()
        })
    elif enc_dist == 'n':
        adj_mat[range(seq_num_nodes), range(seq_num_nodes)] = 1
        # ENT_CLS connect to ent
        adj_mat[0, 2:seq_num_nodes] = adj_mat[2:seq_num_nodes, 0] = 1 - node_types[2:seq_num_nodes]
        # REL_CLS connect to rel
        adj_mat[1, 2:seq_num_nodes] = adj_mat[2:seq_num_nodes, 1] = node_types[2:seq_num_nodes]

        data.update({
            'attention_mask': adj_mat.tolist(),
            'dist_mat': adj_mat.tolist()
        })

    return data


def convert_query_to_adjacent_matrix(query, query_type):
    if query_type in ['2u-DNF', 'up-DNF']:
        if query_type == '2u-DNF':
            queries = [query[0], query[1]]
        elif query_type == 'up-DNF':
            query = tuple2list(query)
            query, rel = query[0], query[1][0]
            query[0][-1].append(rel)
            query[1][-1].append(rel)
            query = list2tuple(query)
            queries = [query[0], query[1]]

        seqs, adjacent_matrixs, layer_seqs, op_seqs = [], [], [], []
        for query in queries:
            seq, adjacent_matrix, layer_seq, op_seq = __convert_query_to_adjacent_matrix(query)
            seqs.append(seq)
            layer_seqs.append(layer_seq)
            adjacent_matrixs.append(adjacent_matrix)
            op_seqs.append(op_seq)

        return seqs, adjacent_matrixs, layer_seqs, op_seqs
    else:
        return __convert_query_to_adjacent_matrix(query)


def __convert_query_to_adjacent_matrix(query):
    num_nodes, edges, node_to_ent, operation_to_node = DagNode.construct_from_query(query)
    layer_to_nodes, layer_to_edge_idxes = topoSort(edges)
    seq = []
    layer_seq = []
    node_idx_to_seq_idx = {}
    for layer in sorted(list(layer_to_nodes.keys())):
        nodes = layer_to_nodes[layer]

        # # ent first, then var
        # nodes = sorted(nodes, key=lambda x: node_to_ent[x] if x in node_to_ent else -1, reverse=True)

        for node in nodes:
            node_idx_to_seq_idx[node] = len(seq)

            if node in node_to_ent:
                seq.append(node_to_ent[node])
            else:
                seq.append(-1)

            layer_seq.append(layer)

    # intersection
    op_seq = [0] * num_nodes
    for op, nodes in operation_to_node.items():
        for node in nodes:
            op_seq[node_idx_to_seq_idx[node]] = op + 1

    seq[-1] = -2
    # consider CLS1 and CLS2
    num_nodes += 2
    adjacent_matrix = [[0] * num_nodes for _ in range(num_nodes)]
    for layer in layer_to_edge_idxes.keys():
        edge_idxes = layer_to_edge_idxes[layer]
        for edge_idx in edge_idxes:
            edge = edges[edge_idx]
            h, r, t = edge
            h, t = node_idx_to_seq_idx[h], node_idx_to_seq_idx[t]

            h, t = h + 2, t + 2

            # 0 means on relation
            adjacent_matrix[h][t] = r + 1
            adjacent_matrix[t][h] = get_inv_rel(r) + 1
    return seq, adjacent_matrix, layer_seq, op_seq


# neg means operation while pos means ent or rel
idx_to_operation = {
    -1: 'union',
    -2: 'neg',
    -3: 'projection',
    -4: 'intersection',
}
operation_to_idx = {v: k for k, v in idx_to_operation.items()}


def is_path(query_structure):
    # whether is rel path
    if query_structure[0] == 'r':
        return True
    else:
        return False


def is_ent(query_structure):
    # whether is anchor node
    if query_structure == 'e':
        return True
    else:
        return False


def is_union(query_structure):
    if query_structure == ('u',):
        return True
    else:
        return False


def convert_query_to_postfix_notation(query, query_structure, DNF=True):
    query_type = query_structure_to_type[query_structure]

    if query_type in ['2p', '3p']:  # , 'pi', 'pni', 'pin']:
        compress_path = True
    else:
        compress_path = False
    # compress_path = False

    return __convert_query_to_postfix_notation(query, query_structure, DNF=DNF, compress_path=compress_path)


def __convert_query_to_postfix_notation(query, query_structure, DNF=True, compress_path=True):
    # converting to DNF
    if DNF:
        if query_structure == (('e', ('r',)), ('e', ('r',)), ('u',)):
            pass
        elif query_structure == ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)):
            query_structure = (('e', ('r', 'r',)), ('e', ('r', 'r')), ('u',))
            query = tuple2list(query)
            query, rel = query[0], query[1][0]
            query[0][-1].append(rel)
            query[1][-1].append(rel)
            query = list2tuple(query)

    postfix_notation = []
    # 0: ent, 1: rel, 2: operation
    type_postfix_notation = []

    if is_ent(query_structure):
        postfix_notation.append(query)
        type_postfix_notation.append(0)

    elif is_path(query_structure):
        # (r, n) -> (-r)
        new_query_structure = []
        for r_type in query_structure:
            if r_type == 'r':
                new_query_structure.append('r')
            elif r_type == 'n':
                new_query_structure[-1] = '-r'

        for r_idx, r_type in zip(query, new_query_structure):
            postfix_notation.append(r_idx)
            type_postfix_notation.append(1)

            if r_type == '-r':
                postfix_notation.append(operation_to_idx['neg'])
                type_postfix_notation.append(operation_to_idx['neg'])

            if not compress_path:
                # r, p, r, p
                postfix_notation.append(operation_to_idx['projection'])
                type_postfix_notation.append(operation_to_idx['projection'])

        if compress_path:
            # r, r, r, p
            postfix_notation.append(operation_to_idx['projection'])
            type_postfix_notation.append(operation_to_idx['projection'])

    elif is_union(query_structure):
        postfix_notation.append(operation_to_idx['union'])
        type_postfix_notation.append(operation_to_idx['union'])

    else:
        all_sub_postfix_notations = []
        all_type_sub_postfix_notations = []
        union_flag = 0
        for idx, (sub_query, sub_structure) in enumerate(zip(query, query_structure)):
            if is_union(sub_structure):
                # ('u', )
                union_flag = 1
                continue

            sub_postfix_notation, sub_type_postfix_notation = \
                __convert_query_to_postfix_notation(sub_query, sub_structure, DNF, compress_path)

            if is_path(sub_structure):
                # [e] + [r, r, p] -> [e, r, r, p]
                all_sub_postfix_notations[-1].extend(sub_postfix_notation)
                all_type_sub_postfix_notations[-1].extend(sub_type_postfix_notation)

            else:
                all_sub_postfix_notations.append(sub_postfix_notation)
                all_type_sub_postfix_notations.append(sub_type_postfix_notation)

        if len(all_sub_postfix_notations) > 1:
            # intersection or union
            for sub_postfix_notation, sub_type_postfix_notation in zip(all_sub_postfix_notations,
                                                                       all_type_sub_postfix_notations):
                # # [e, r, p]-> [e, r], [e, r, r, p] -> [e, r, r, p]
                # num_hop = 0
                # for r in sub_type_postfix_notation:
                #     # rel or neg
                #     if r == 1:
                #         num_hop += 1
                # if num_hop == 1:
                sub_postfix_notation = sub_postfix_notation[:-1]
                sub_type_postfix_notation = sub_type_postfix_notation[:-1]

                postfix_notation.extend(sub_postfix_notation)
                type_postfix_notation.extend(sub_type_postfix_notation)

            operation = 'union' if union_flag else 'intersection'
            postfix_notation.append(operation_to_idx[operation])
            type_postfix_notation.append(operation_to_idx[operation])
        else:
            postfix_notation.extend(all_sub_postfix_notations[0])
            type_postfix_notation.extend(all_type_sub_postfix_notations[0])

        # for sub_postfix_notation, sub_type_postfix_notation in zip(all_sub_postfix_notations,
        #                                                            all_type_sub_postfix_notations):
        #     postfix_notation.extend(sub_postfix_notation)
        #     type_postfix_notation.extend(sub_type_postfix_notation)
        #
        # if len(all_sub_postfix_notations) > 1 and postfix_notation[-1] != operation_to_idx['union']:
        #     postfix_notation.append(operation_to_idx['intersection'])
        #     type_postfix_notation.append(operation_to_idx['intersection'])

    return postfix_notation, type_postfix_notation


if __name__ == '__main__':
    train_queries = pickle.load(open(os.path.join('data/FB15k-237-betae', "valid-queries.pkl"), 'rb'))

    for query_struct in train_queries.keys():
        if query_struct not in query_structure_to_type:
            continue
        q = train_queries[query_struct].pop()

        print(query_struct)
        print(q)
        data = encode_query(q, query_structure_to_type[query_struct], 'd')
        print('x:         ', data['x'])
        print('layers:    ', data['layers'])
        print('types:     ', data['node_types'])
        print('negs:      ', data['negs'])
        print('in:        ', data['in_degs'])
        print('out:       ', data['out_degs'])
        print('tgt_type:    ', data['tgt_type'])
        print('attn_mask: ', data['attention_mask'])
        print(np.array(data['dist_mat']))
        print(data['query_type'])
        # seq, adjacent_matrix, layer_seq, op_seq = convert_query_to_adjacent_matrix(q, query_structure_to_type[k])
        # print(seq)
        # print(layer_seq)
        # print(op_seq)
        # for i in adjacent_matrix:
        #     print(i)
        # print()
