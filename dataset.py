#!/usr/bin/python3

import itertools
import json
import random
import time
from collections import defaultdict
from typing import List, Optional

import logging
import numpy as np
import pickle

from pathlib import Path
import os
import torch

from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.loader.dataloader import Collater
from tqdm import tqdm

from utils import flatten, list2tuple, tuple2list
from query_format_converter import convert_query_to_pyg, get_inv_rel, convert_query_to_postfix_notation, \
    convert_query_to_adjacent_matrix, encode_query

from utils import DEV, RE_ENC

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
                           ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                           ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
                           }
query_type_to_structure = {value: key for key, value in query_structure_to_type.items()}
all_tasks = list(
    query_type_to_structure.keys())  # ['1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2in', '3in', 'inp', 'pin', 'pni', '2u-DNF', '2u-DM', 'up-DNF', 'up-DM']

graph_collate_fn = Collater(None, None)


def flatten_query(queries):
    all_queries = []
    for query_structure in queries:
        tmp_queries = list(queries[query_structure])
        all_queries.extend([(query, query_structure_to_type[query_structure]) for query in tmp_queries])
    return all_queries


def load_data(args, tasks):
    '''
    Load queries and remove queries not in tasks
    '''
    logging.info("loading data")

    train_queries = pickle.load(open(os.path.join(args.data_path, "train-queries.pkl"), 'rb'))
    train_answers = pickle.load(open(os.path.join(args.data_path, "train-answers.pkl"), 'rb'))

    valid_queries = pickle.load(open(os.path.join(args.data_path, "valid-queries.pkl"), 'rb'))
    valid_hard_answers = pickle.load(open(os.path.join(args.data_path, "valid-hard-answers.pkl"), 'rb'))
    valid_easy_answers = pickle.load(open(os.path.join(args.data_path, "valid-easy-answers.pkl"), 'rb'))
    test_queries = pickle.load(open(os.path.join(args.data_path, "test-queries.pkl"), 'rb'))
    test_hard_answers = pickle.load(open(os.path.join(args.data_path, "test-hard-answers.pkl"), 'rb'))
    test_easy_answers = pickle.load(open(os.path.join(args.data_path, "test-easy-answers.pkl"), 'rb'))

    # remove tasks not in args.tasks
    for name in all_tasks:
        if 'u' in name:
            name, evaluate_union = name.split('-')
        else:
            evaluate_union = args.evaluate_union
        if name not in tasks or evaluate_union != args.evaluate_union:
            query_structure = query_type_to_structure[name if 'u' not in name else '-'.join([name, evaluate_union])]
            if query_structure in train_queries:
                del train_queries[query_structure]
            if query_structure in valid_queries:
                del valid_queries[query_structure]
            if query_structure in test_queries:
                del test_queries[query_structure]

    return train_queries, train_answers, valid_queries, valid_hard_answers, valid_easy_answers, test_queries, test_hard_answers, test_easy_answers


class TestDataset(Dataset):
    def __init__(self, queries, num_ent, num_rel, bi_dir=True):
        # queries is a list of (query, query_structure) pairs
        self.len = len(queries)
        self.queries = queries
        self.num_ent = num_ent
        self.num_rel = num_rel

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        query = self.queries[idx][0]
        query_structure = self.queries[idx][1]
        negative_sample = torch.LongTensor(range(self.num_ent))
        return negative_sample, flatten(query), query, query_structure

    @staticmethod
    def collate_fn(data):
        negative_sample = torch.stack([_[0] for _ in data], dim=0)
        query = [_[1] for _ in data]
        query_unflatten = [_[2] for _ in data]
        query_structure = [_[3] for _ in data]
        return negative_sample, query, query_unflatten, query_structure


class TrainDataset(Dataset):
    def __init__(self, queries, num_ent, num_rel, negative_sample_size, answer, data_path, bi_dir=True):
        # queries is a list of (query, query_structure) pairs
        self.queries = queries
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.negative_sample_size = negative_sample_size
        self.count = self.count_frequency(queries, answer)
        self.answer = answer
        self.data_path = data_path

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx][0]
        query_structure = self.queries[idx][1]
        tail = np.random.choice(list(self.answer[query]))
        subsampling_weight = self.count[query]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.num_ent, size=self.negative_sample_size * 2)
            mask = np.in1d(
                negative_sample,
                self.answer[query],
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.from_numpy(negative_sample)
        positive_sample = torch.LongTensor([tail])
        return positive_sample, negative_sample, subsampling_weight, flatten(query), query_structure

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.cat([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        query = [_[3] for _ in data]
        query_structure = [_[4] for _ in data]
        return positive_sample, negative_sample, subsample_weight, query, query_structure

    @staticmethod
    def count_frequency(queries, answer, start=4):
        count = {}
        for query, qtype in queries:
            count[query] = start + len(answer[query])
        return count


def merge_graph(graphs: List[Data]):
    edge_index = []
    edge_type = []
    operation_to_node = defaultdict(list)
    node_to_ent = {}
    inter_node_to_ent = {}
    target_to_in_edges = {}
    target_node_idxes = []
    query_idx_to_union_nodes = {}
    layer_to_nodes = defaultdict(list)
    layer_to_edge_idxes = defaultdict(list)
    query_idxes = []
    layer_to_tail_to_edge_idxes = defaultdict(dict)

    num_nodes = 0
    num_edges = 0

    keys = graphs[0].keys

    for graph in graphs:
        for key in keys:
            if key == 'edge_index':
                edge_index.append(graph[key] + num_nodes)
            elif key == 'edge_type':
                edge_type.append(graph[key])
            elif key == 'node_to_ent':
                for node, ent in graph[key].items():
                    node_to_ent[node + num_nodes] = ent
            elif key == 'inter_node_to_ent':
                for node, ent in graph[key].items():
                    inter_node_to_ent[node + num_nodes] = ent
            elif key == 'layer_to_tail_to_edge_idxes':
                for layer, tail_to_edge_idxes in graph[key].items():
                    for tail, edge_idxes in tail_to_edge_idxes.items():
                        layer_to_tail_to_edge_idxes[layer][tail + num_nodes] = [idx + num_edges for idx in edge_idxes]
            elif key == 'operation_to_node':
                for operation, nodes in graph[key].items():
                    operation_to_node[operation].extend([idx + num_nodes for idx in nodes])
            elif key == 'target_to_in_edges':
                for tail, edges in graph[key].items():
                    target_to_in_edges[tail + num_nodes] = edges
            elif key == 'layer_to_nodes':
                for layer, nodes in graph[key].items():
                    layer_to_nodes[layer].extend([idx + num_nodes for idx in nodes])
            elif key == 'layer_to_edge_idxes':
                for layer, idxes in graph[key].items():
                    layer_to_edge_idxes[layer].extend([idx + num_edges for idx in idxes])

        if len(graph['query_idx_to_union_nodes']) != 0:
            for query_idx, union_nodes_idx in graph['query_idx_to_union_nodes'].items():
                union_nodes_idx = [idx + num_nodes for idx in union_nodes_idx]
                # get the idx in current batch
                query_idx_to_union_nodes[query_idx] = union_nodes_idx
        else:
            target_node_idxes.append(graph['target_node_idxes'] + num_nodes)

        if 'query_idx' in keys:
            query_idxes.append(graph['query_idx'])

        num_nodes += graph.num_nodes
        num_edges += graph.num_edges_

    edge_index = np.concatenate(edge_index, axis=1)
    edge_type = np.concatenate(edge_type)

    graph = Data(edge_index=edge_index)

    graph.edge_type = edge_type
    graph.node_to_ent = node_to_ent
    graph.inter_node_to_ent = inter_node_to_ent
    graph.operation_to_node = operation_to_node
    graph.num_nodes = num_nodes
    graph.num_edges_ = num_edges
    graph.target_node_idxes = target_node_idxes
    graph.target_to_in_edges = target_to_in_edges
    graph.layer_to_nodes = layer_to_nodes
    graph.layer_to_edge_idxes = layer_to_edge_idxes
    graph.query_idx_to_union_nodes = query_idx_to_union_nodes
    graph.query_idxes = query_idxes
    graph.layer_to_tail_to_edge_idxes = layer_to_tail_to_edge_idxes

    return graph


class DagTrainDataset(TrainDataset):
    def __init__(self, queries, num_ent, num_rel, negative_sample_size, answer, data_path, bi_dir=False,
                 use_pseudo_labels=False, mask_strategy=0, query_types=None, random_chose_tail=True):
        """
        mask_strategy:
            0: not use mask strategy;
            1: random mask node;
            2: only mask intermediate node (predict pseudo label by P-Net)
        """
        # queries is a list of (query, query_structure) pairs
        super().__init__(queries, num_ent, num_rel, negative_sample_size, answer, data_path)

        # random chose an answer from candidate answers in __getitem__
        # if random_chose_tail is False, query will repeat n times, where n is the number of candidate answers
        self.random_chose_tail = random_chose_tail
        if not self.random_chose_tail:
            new_queries = []
            for query in self.queries:
                query, structure = query
                for tail in self.answer[query]:
                    new_queries.append((query, structure, tail))
            self.queries = new_queries

        # e.g. 1p,2p,3p
        self.query_types = query_types

        # is mask_strategy==True, mask mode will be used, which is the strategy used by P-Net
        self.mask_strategy = mask_strategy
        self.use_pseudo_labels = use_pseudo_labels

        self.bi_dir = bi_dir

        if DEV:
            self.query_to_pyg = None
        else:
            self.query_to_pyg = {}
            logging.info('pre converting to pyg...')
            for query in tqdm(queries):
                query = query[0]
                pyg = convert_query_to_pyg(query, self.bi_dir)
                self.query_to_pyg[query] = pyg

    def __getitem__(self, idx):
        # print(f"id: {id(self)}, use_pseudo_labels: {self.use_pseudo_labels}, "
        #       f"mask_strategy: {self.mask_strategy}, "
        #       f"bi_dir:  {self.bi_dir}")
        query = self.queries[idx][0]
        query_structure = self.queries[idx][1]
        if self.random_chose_tail:
            tail = np.random.choice(list(self.answer[query]))
        else:
            tail = self.queries[idx][2]

        if self.query_to_pyg:
            query_pyg = self.query_to_pyg[query]
        else:
            query_pyg = convert_query_to_pyg(query, bi_dir=False)

        query_pyg.query_idx = idx

        # if self.use_pseudo_labels and query_name_dict[query_structure] in ['2p', '3p']:
        #     # print(len(QUERY_TO_INTER_NODE_TO_ENTITY))
        #     # update intermediate node entities, which is obtained by Q or P net
        #
        #     # if DEV:
        #     #     if query not in QUERY_TO_INTER_NODE_TO_ENTITY:
        #     #         inter_node_to_ent = query_pyg.inter_node_to_ent
        #     #         for k, v in inter_node_to_ent.items():
        #     #             if inter_node_to_ent[k] == -1:
        #     #                 inter_node_to_ent[k] = random.randrange(0, self.num_ent)
        #     # else:
        #     #     query_pyg.inter_node_to_ent = QUERY_TO_INTER_NODE_TO_ENTITY[query]
        #     pass
        # if self.mask_strategy == 1:
        #     # get ent of each node
        #     query_pyg.node_to_ent = {**query_pyg.inter_node_to_ent, **query_pyg.node_to_ent}
        #     query_pyg.node_to_ent[query_pyg.target_node_idxes] = tail
        #
        #     # delete unlabeled intermediate node, e.g. 1: -1
        #     query_pyg.node_to_ent = {k: v for k, v in query_pyg.node_to_ent.items() if v != -1}
        #
        #     # use node chose randomly replace the origin target node
        #     query_pyg.target_node_idxes = random.choice(list(query_pyg.node_to_ent.keys()))
        #     # pop the masked node as positive sample
        #     tail = query_pyg.node_to_ent.pop(query_pyg.target_node_idxes)
        # elif self.mask_strategy == 2:
        #     # no target node is needed, as this is predicting intermediate nodes
        #     query_pyg.node_to_ent[query_pyg.target_node_idxes] = tail

        subsampling_weight = self.count[query]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.num_ent, size=self.negative_sample_size * 2)
            mask = np.in1d(
                negative_sample,
                self.answer[query],
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.from_numpy(negative_sample)
        positive_sample = torch.LongTensor([tail])
        return positive_sample, negative_sample, subsampling_weight, query_pyg, query_structure

    @staticmethod
    def collate_fn_return_all_graphs(data):
        positive_sample = torch.cat([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        graphs = [_[3] for _ in data]
        final_graph = merge_graph(graphs)

        final_graph.edge_index = torch.LongTensor(final_graph.edge_index)
        final_graph.edge_type = torch.LongTensor(final_graph.edge_type)

        query_structures = [_[4] for _ in data]

        return positive_sample, negative_sample, subsample_weight, final_graph, query_structures, graphs

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.cat([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        graphs = [_[3] for _ in data]
        final_graph = merge_graph(graphs)

        final_graph.edge_index = torch.LongTensor(final_graph.edge_index)
        final_graph.edge_type = torch.LongTensor(final_graph.edge_type)

        query_structures = [_[4] for _ in data]

        return positive_sample, negative_sample, subsample_weight, final_graph, query_structures


class DagTestDataset(Dataset):
    def __init__(self, queries, num_ent, num_rel, bi_dir=True):
        # queries is a list of (query, query_structure) pairs
        self.bi_dir = bi_dir

        self.len = len(queries)
        self.queries = queries
        self.num_ent = num_ent
        self.num_rel = num_rel

        # logging.info('converting queries to pyg')
        # self.query_pygs = [convert_query_to_pyg(query[0]) for query in self.queries]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        query = self.queries[idx][0]
        query_type = self.queries[idx][1]

        if 'DNF' in query_type:
            # convert query in DNF

            if query_type[:2] == 'up':
                # (((9590, (62,)), (159, (62,)), (-1,)), (4,))
                l_query = tuple2list(query)
                l_query, r = l_query

                # [[9590, [62,]], [159, [62,]]
                q1, q2 = l_query[0], l_query[1]

                # [[9590, [62, 4]], [159, [62, 4]]
                q1[1].extend(r)
                q2[1].extend(r)
                query_idx_to_union_nodes = (0, 3)

            elif query_type[:2] == '2u':
                l_query = tuple2list(query)
                q1, q2 = l_query[0], l_query[1]
                query_idx_to_union_nodes = (0, 2)

            else:
                assert False, "unknown DNF type"

            q1 = list2tuple(q1)
            q2 = list2tuple(q2)

            pyg1 = convert_query_to_pyg(q1, self.bi_dir)
            pyg2 = convert_query_to_pyg(q2, self.bi_dir)

            # merge two 2p into one graph
            query_pyg = merge_graph([pyg1, pyg2])
            query_pyg.query_idx_to_union_nodes[idx] = query_idx_to_union_nodes
        else:
            query_pyg = convert_query_to_pyg(query, self.bi_dir)

        query_pyg.query_idx = idx

        negative_sample = torch.LongTensor(range(self.num_ent))
        return negative_sample, query_pyg, query, query_type

    @staticmethod
    def collate_fn(data):
        negative_sample = torch.stack([_[0] for _ in data], dim=0)
        query = [_[1] for _ in data]

        final_graph = merge_graph(query)

        final_graph.edge_index = torch.LongTensor(final_graph.edge_index)
        final_graph.edge_type = torch.LongTensor(final_graph.edge_type)

        query_unflatten = [_[2] for _ in data]
        query_structure = [_[3] for _ in data]
        return negative_sample, final_graph, query_unflatten, query_structure


class Q2TTrainDataset(TrainDataset):
    def __init__(self, queries, num_ent, num_rel, negative_sample_size, answer, data_path,
                 enc_dist='d'):
        super().__init__(queries, num_ent, num_rel, negative_sample_size, answer, data_path)

        self.query_to_graph = None
        self.enc_dist = enc_dist
        save_path = Path(data_path) / 'encoded_train_queries.pkl'

        if not DEV:
            self.query_to_graph = {}
            if save_path.exists() and not RE_ENC:
                with open(save_path, 'rb') as f:
                    logging.info(f'loading encoded queries from {save_path}...')
                    self.query_to_graph = pickle.load(f)
                self.query_to_graph = {query: self.query_to_graph[query] for query, _ in self.queries}
            else:
                logging.info('pre encoding queries...')
                for query in tqdm(queries):
                    query, query_type = query
                    query_structure = query_type_to_structure[query_type]

                    graph = encode_query(query, query_type)
                    self.query_to_graph[query] = graph

                with open(save_path, 'wb') as f:
                    logging.info(f'saving encoded queries to {save_path}')
                    pickle.dump(self.query_to_graph, f)

    def __getitem__(self, idx):
        query = self.queries[idx][0]
        query_type = self.queries[idx][1]
        query_structure = query_type_to_structure[query_type]

        tail = np.random.choice(list(self.answer[query]))

        if self.query_to_graph:
            batch = self.query_to_graph[query]
        else:
            batch = encode_query(query, query_type, self.enc_dist)

        for k in batch:
            batch[k] = torch.LongTensor(batch[k])

        subsampling_weight = 1
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size:
            negative_samples = np.random.randint(self.num_ent, size=self.negative_sample_size * 2)
            mask = np.in1d(
                negative_samples,
                self.answer[query],
                assume_unique=True,
                invert=True
            )
            negative_samples = negative_samples[mask]
            negative_sample_list.append(negative_samples)
            negative_sample_size += negative_samples.size
        negative_samples = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_samples = torch.from_numpy(negative_samples)
        positive_samples = torch.LongTensor([tail])

        batch.update({
            'positive_samples': positive_samples,
            'negative_samples': negative_samples
        })
        return batch

    @staticmethod
    def collate_fn(data):
        batch_queries = data
        elem = data[0]
        batch = {
            key: torch.stack([data[key] for data in batch_queries]).squeeze()
            for key in elem
        }

        return batch


class Q2TTestDataset(Dataset):
    def __init__(self, queries, num_ent, num_rel, enc_dist):
        # queries is a list of (query, query_structure) pairs
        self.len = len(queries)
        self.queries = queries
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.enc_dist = enc_dist

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        query = self.queries[idx][0]
        query_type = self.queries[idx][1]
        query_structure = query_type_to_structure[query_type]

        batch = encode_query(query, query_type, self.enc_dist)
        for k in batch:
            batch[k] = torch.LongTensor(batch[k])

        batch['negative_samples'] = torch.LongTensor(range(self.num_ent))

        return query, query_type, batch

    @staticmethod
    def collate_fn(data):
        query_unflatten = [_[0] for _ in data]
        query_types = [_[1] for _ in data]

        batch_queries = [_[2] for _ in data]
        elem = batch_queries[0]
        batch = {
            key: torch.stack([data[key] for data in batch_queries]).squeeze()
            for key in elem
        }

        return query_unflatten, query_types, batch


class SingledirectionalOneShotIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)
        self.step = 0

    def __next__(self):
        self.step += 1
        data = next(self.iterator)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data
