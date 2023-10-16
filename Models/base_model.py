import collections
import logging
import time

import torch
from torch import nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing, MLP
from tqdm import tqdm

from Models.utils import init_weights
from utils import calc_metric_and_record


class Predict(MessagePassing):
    def __init__(self, num_ents, num_rels, hidden_dim, **kwargs):
        super().__init__()
        self.kwargs = kwargs

        self.dim_ent_embedding = kwargs['de'] if kwargs['de'] else hidden_dim
        self.dim_rel_embedding = kwargs['dr'] if kwargs['dr'] else hidden_dim

        self.hidden_dim = hidden_dim
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.attn_lin1 = MLP(in_channels=hidden_dim, hidden_channels=hidden_dim,
                             out_channels=hidden_dim, num_layers=2, norm=None)

        # self.reset_parameters()

    def reset_parameters(self):
        self.attn_lin1.apply(init_weights)


class GNNBasedModel(nn.Module):
    def __init__(self, num_ents, num_rels, hidden_dim, edge_to_entities,
                 cuda=True, model_name=None, model_type='p', **kwargs):
        super(GNNBasedModel, self).__init__()

        self.model_name = model_name
        self.model_type = model_type

        self.num_ent = num_ents
        self.num_rel = num_rels
        self.edge_to_entities = edge_to_entities

        self.epsilon = 2.0
        gamma = kwargs['gamma']
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.dim_ent_embedding = kwargs['de'] if kwargs['de'] else hidden_dim
        self.dim_rel_embedding = kwargs['dr'] if kwargs['dr'] else hidden_dim

        self.ent_embedding = nn.Embedding(num_ents, self.dim_ent_embedding)
        self.rel_embedding = nn.Embedding(num_rels, self.dim_rel_embedding)

        for t in [self.ent_embedding, self.rel_embedding]:
            nn.init.uniform_(
                tensor=t.weight,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

        self.lambda_ = nn.Parameter(torch.Tensor([kwargs['lambda']]), requires_grad=False)

        self.device = torch.device('cuda') if cuda else torch.device('cpu')

        self.hidden_dim = hidden_dim
        self.p_dropout = kwargs['p_dropout']

        # calc_pseudo_label_loss will be set to True in the beginning of iterative training
        self.calc_pseudo_label_loss = False
        self.record_weights = kwargs['record_weights']

        self.fp16 = kwargs['fp16']
        if self.fp16:
            self.scaler = GradScaler()
        else:
            self.scaler = None

    def calc_logit(self, x, target_node_idxes,
                   query_idx_to_union_nodes=None,
                   positive_samples=None, negative_samples=None):
        if len(x.shape) == 2:
            positive_logit, negative_logit = self.__calc_logit(x, target_node_idxes,
                                                               query_idx_to_union_nodes=query_idx_to_union_nodes,
                                                               positive_samples=positive_samples,
                                                               negative_samples=negative_samples)
        else:
            # has n gaussian
            positive_logits, negative_logits = [], []
            for i in range(x.shape[1]):
                positive_logit, negative_logit = self.__calc_logit(x[:, i, :], target_node_idxes,
                                                                   query_idx_to_union_nodes=query_idx_to_union_nodes,
                                                                   positive_samples=positive_samples,
                                                                   negative_samples=negative_samples)
                positive_logits.append(positive_logit)
                negative_logits.append(negative_logit)

            if positive_logits[0] is not None:
                # positive_logit is None in test mode
                positive_logit = torch.stack(positive_logits, dim=-1)
                positive_logit, _ = positive_logit.max(dim=-1)
            else:
                positive_logit = None

            negative_logit = torch.stack(negative_logits, dim=-1)
            negative_logit, _ = negative_logit.max(dim=-1)

        return positive_logit, negative_logit

    def __calc_logit(self, x, target_node_idxes,
                     query_idx_to_union_nodes=None,
                     positive_samples=None, negative_samples=None):
        """
        only in test mode, query_idx_to_union_nodes won't be None
        """
        # negative_sample won't be None
        num_query = negative_samples.shape[0]

        if query_idx_to_union_nodes is None:
            # train mode, there isn't union query
            conj_query_idxes = list(range(num_query))

            pred_embeddings = x[target_node_idxes].unsqueeze(1)

            positive_embedding = self.ent_embedding(positive_samples).unsqueeze(1)
            positive_logit = self.gamma - torch.norm(positive_embedding - pred_embeddings, p=1, dim=-1)

            negative_embedding = self.ent_embedding(negative_samples)
            negative_logit = self.gamma - torch.norm(negative_embedding - pred_embeddings, p=1, dim=-1)

            return positive_logit, negative_logit
        else:
            # test mode, there is only negative logit
            disj_query_idxes = query_idx_to_union_nodes.keys()
            conj_query_idxes = list(set(range(num_query)) - set(disj_query_idxes))

            negative_logit = torch.zeros(negative_samples.shape, device=self.device)

            conj_pred_embeddings = x[target_node_idxes].unsqueeze(1)
            conj_positive_embedding = self.ent_embedding(negative_samples[conj_query_idxes])
            negative_logit[conj_query_idxes] = self.gamma - torch.norm(conj_positive_embedding - conj_pred_embeddings,
                                                                       p=1, dim=-1)

            for disj_query_idx, union_node_idxes in query_idx_to_union_nodes.items():
                pred_embeddings = x[union_node_idxes].unsqueeze(1)
                negative_embedding = self.ent_embedding(negative_samples[disj_query_idx])
                # [num_union, num_neg]
                union_logits = self.gamma - torch.norm(negative_embedding - pred_embeddings, p=1, dim=-1)
                negative_logit[disj_query_idx] = torch.max(union_logits, dim=0)[0]

            return None, negative_logit

        # pred_embeddings = x[target_node_idxes].unsqueeze(1)
        #
        # positive_logit, negative_logit = None, None
        #
        # if type(positive_sample) != type(None):
        #     if len(pred_embeddings) > 0:
        #         positive_embedding = self.ent_embedding(positive_sample).unsqueeze(1)
        #         positive_logit = self.gamma - torch.norm(positive_embedding - pred_embeddings, p=1, dim=-1)
        #
        # if type(negative_sample) != type(None):
        #     if len(pred_embeddings) > 0:
        #         # batch_size, num_neg, dim
        #         negative_embedding = self.ent_embedding(negative_sample)
        #         negative_logit = self.gamma - torch.norm(negative_embedding - pred_embeddings, p=1, dim=-1)
        #
        # return positive_logit, negative_logit

    def forward(self, pyg):
        raise NotImplementedError

    def get_post_prb(self, pyg, target_node_ent):
        raise NotImplementedError

    def train_step(self: 'GNNBasedModel', optimizer, train_iterator, args, query_type):
        """

        :param self:
        :param optimizer:
        :param train_iterator:
        :param args:
        :param model2: used for pseudo label predicting
        :return:
        """
        self.train()
        optimizer.zero_grad()

        t1 = time.time()

        positive_samples, negative_sample, subsampling_weight, query_pyg, query_structures = next(train_iterator)

        # t2 = time.time()
        # print('loading ', t2 - t1)
        # t1 = t2

        if args.cuda:
            positive_samples = positive_samples.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
            query_pyg.edge_type = query_pyg.edge_type.cuda()
            query_pyg.edge_index = query_pyg.edge_index.cuda()

        # if model2:
        #     query_pyg = _update_inter_node_to_ent_in_graph(model2, positive_samples, query_pyg)

        with autocast(enabled=args.fp16):
            x, weights = self(query_pyg)

            positive_logit, negative_logit = self.calc_logit(x, query_pyg.target_node_idxes,
                                                             positive_samples=positive_samples,
                                                             negative_samples=negative_sample)

            log, loss = calc_loss(negative_logit, positive_logit, subsampling_weight)

            if args.record_weights:
                structure_to_idxes = {k: [] for k in set(query_structures)}
                for i, structure in enumerate(query_structures):
                    structure_to_idxes[structure].append(i)

                structure_to_weights = {structure: weights[idxes].mean(dim=0).cpu().detach().numpy() for
                                        structure, idxes in
                                        structure_to_idxes.items()}
                log['structure_to_weights'] = structure_to_weights

            if self.calc_pseudo_label_loss and len(query_pyg.inter_node_to_ent):
                inter_node_to_ent = query_pyg.inter_node_to_ent
                inter_node_idxes, inter_node_entities = list(inter_node_to_ent.keys()), list(
                    inter_node_to_ent.values())

                inter_node_entities = torch.tensor(inter_node_entities, dtype=torch.long).cuda()

                logit = self.gamma - torch.norm(self.ent_embedding(inter_node_entities) - x[inter_node_idxes],
                                                p=1, dim=-1)
                inter_loss = -F.logsigmoid(logit).mean()
                inter_loss = inter_loss * self.lambda_
                loss = loss + inter_loss

                # add and change loss in log
                log['inter_loss'] = inter_loss.item()
                log['loss'] = loss.item()

        if self.fp16:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()

        return log

    def test_step(self, easy_answers, hard_answers, args, query_type_to_iterator, query_name_dict, save_result=False,
                  save_str="", save_empty=False):
        self.eval()

        step = 0
        total_steps = sum([len(iterator) for iterator in query_type_to_iterator.values()])
        logs = collections.defaultdict(list)

        all_structure_to_weights = []

        with torch.no_grad():
            for query_type, iterator in query_type_to_iterator.items():
                for negative_sample, query_pyg, queries_unflatten, query_structures in tqdm(iterator):
                    if args.cuda:
                        negative_sample = negative_sample.cuda()
                        query_pyg.to(self.device)

                    x, weights = self(query_pyg)

                    min_query_idx = min(query_pyg.query_idxes)
                    # get the idx in current batch
                    query_pyg.query_idx_to_union_nodes = {k - min_query_idx: v for k, v in
                                                          query_pyg.query_idx_to_union_nodes.items()}

                    positive_logit, negative_logit = self.calc_logit(x, query_pyg.target_node_idxes,
                                                                     query_idx_to_union_nodes=query_pyg.query_idx_to_union_nodes,
                                                                     positive_samples=None,
                                                                     negative_samples=negative_sample)

                    tmp_logs, tmp_records = calc_metric_and_record(args, easy_answers, hard_answers, negative_logit,
                                                                   queries_unflatten,
                                                                   query_structures, x)
                    for query_structure, res in tmp_logs.items():
                        logs[query_structure].extend(res)

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

                    if args.record_weights:
                        # record weights
                        structure_to_idxes = {k: [] for k in set(query_structures)}
                        for i, structure in enumerate(query_structures):
                            structure_to_idxes[structure].append(i)

                        structure_to_weights = {structure: weights[idxes].mean(dim=0) for structure, idxes in
                                                structure_to_idxes.items()}
                        all_structure_to_weights.append(structure_to_weights)

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]]) / len(
                    logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        if args.record_weights:
            # record weights
            structure_to_weights = collections.defaultdict(list)
            for dict_ in all_structure_to_weights:
                for k, v in dict_.items():
                    structure_to_weights[k].append(v)
            structure_to_weights = {k: torch.stack(v).mean(dim=0).tolist() for k, v in structure_to_weights.items()}

            metrics['structure_to_weights'] = structure_to_weights

        return metrics

    def sample_from_edge_type(self, edges, idxes):
        candidate_entities = []
        for edge in edges:
            candidate_entities.extend(self.edge_to_entities[edge])

        entities = [candidate_entities[idx % len(candidate_entities)] for idx in idxes]

        return entities


def calc_loss(negative_logit, positive_logit, subsampling_weight):
    negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
    positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)
    positive_sample_loss = - (subsampling_weight * positive_score).sum()
    negative_sample_loss = - (subsampling_weight * negative_score).sum()
    positive_sample_loss /= subsampling_weight.sum()
    negative_sample_loss /= subsampling_weight.sum()
    loss = (positive_sample_loss + negative_sample_loss) / 2

    log = {
        'positive_sample_loss': positive_sample_loss.item(),
        'negative_sample_loss': negative_sample_loss.item(),
        'loss': loss.item(),
    }

    return log, loss


def index_by_list(a, b):
    # index a by b
    idxes = []
    ptr1, ptr2 = 0, 0
    while ptr1 < len(a) and ptr2 < len(b):
        if a[ptr1] == b[ptr2]:
            idxes.append(ptr2)
            ptr1 += 1
            ptr2 += 1
        elif a[ptr1] < b[ptr2]:
            ptr1 += 1
        else:
            ptr2 += 1

    return idxes
