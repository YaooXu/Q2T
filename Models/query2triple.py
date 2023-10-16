import collections
import logging
import time
import os

import math
import numpy as np
import torch
from torch import nn as nn, Tensor
from torch.cuda.amp import autocast, GradScaler
from torch.nn import ModuleList, Linear, BatchNorm1d
from torch.nn.init import xavier_normal_, kaiming_uniform_, kaiming_normal_
from torch_geometric.data import Data
from transformers import BertConfig
from torch_geometric.nn import MLP, RGCNConv
from .modeling_bert import BertModel
from .transformer_conv import TransformerConv

from tqdm import tqdm
from utils import OFFSET, REL_CLS, ENT_CLS, TYPE_TO_IDX, \
    TYPE_TO_SMOOTH, expand, calc_metric_and_record

import torch.nn.functional as F


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1,
                 reduction="mean"):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
            if self.reduction == 'sum' else loss

    def linear_combination(self, x, y, smoothing=None):
        if smoothing is None:
            smoothing = self.smoothing

        return smoothing * x + (1 - smoothing) * y

    def forward(self, preds, target, query_types=None):
        if query_types is not None:
            smoothing = torch.ones(query_types.shape, device=query_types.device)
            for type_ in TYPE_TO_SMOOTH.keys():
                idx = TYPE_TO_IDX[type_]
                smoothing[query_types == idx] = TYPE_TO_SMOOTH[type_]
        else:
            assert 0 <= self.smoothing < 1
            smoothing = self.smoothing

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1)) / n
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction
        )
        return self.linear_combination(loss, nll, smoothing), log_preds


class Query2Triple(nn.Module):
    def __init__(self, num_ents, num_rels, hidden_dim, edge_to_entities,
                 **kwargs):
        super(Query2Triple, self).__init__()

        # basic setting
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.edge_to_entities = edge_to_entities
        self.device = torch.device('cuda') if kwargs['cuda'] else torch.device('cpu')

        self.dim_ent_embedding = kwargs['dim_ent_embedding']
        self.dim_rel_embedding = kwargs['dim_rel_embedding']

        model = {
            'tucker': TuckER,
            'complex': Complex,
            'cp': CP,
            'rescal': RESCAL,
            'distmult': DistMult
        }[kwargs['geo'].lower()]
        self.kge_model = model(self.num_ents, self.num_rels,
                               self.dim_ent_embedding, self.dim_rel_embedding,
                               kwargs)

        self.kge_ckpt_path = kwargs['kge_ckpt_path']
        if self.kge_ckpt_path:
            self.kge_model.load_from_ckpt_path(self.kge_ckpt_path)

        self.ent_embedding = self.kge_model.ent_embedding
        self.rel_embedding = self.kge_model.rel_embedding
        kge_requires_grad = True if kwargs['not_freeze_kge'] else False

        if not kge_requires_grad:
            for name, param in self.kge_model.named_parameters():
                param.requires_grad = False

        logging.info(f'KGE requires_grad: {kge_requires_grad}')

        if self.kge_ckpt_path and not kge_requires_grad:
            self.use_kge_to_pred_1p = True
        else:
            self.use_kge_to_pred_1p = False
        logging.info(f'use_kge_to_pred_1p: {self.use_kge_to_pred_1p}')

        # var, tgt, CLS
        self.sp_token_embedding = nn.Embedding(OFFSET, self.dim_ent_embedding)
        # prompt
        self.query_encoder = BertEncoder(kwargs)

        self.fp16 = kwargs['fp16']
        if self.fp16:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        self.loss_fct = LabelSmoothingLoss(smoothing=kwargs['label_smoothing'], reduction='none')

        self.init_weight()

    def init_weight(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding) and module.weight.requires_grad:
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def init_seq_embedding(self, seq: torch.tensor):
        """
        :param seq:
        :param embedding1:
        :param embedding2:
        :return:
        """
        #  init sqe embedding
        shape = seq.shape + (self.ent_embedding.embedding_dim,)
        node_embeddings = torch.zeros(shape, device=self.device)

        # ent idx
        tup_idx = torch.where(seq >= OFFSET)
        node_embeddings[tup_idx] = self.ent_embedding(seq[tup_idx] - OFFSET)

        # 0: var, 1: tgt, 2: ENT_CLS, 3: REL_CLS
        tup_idx = torch.where((seq >= 0) & (seq < OFFSET))
        node_embeddings[tup_idx] = self.sp_token_embedding(seq[tup_idx])

        # rel idx
        tup_idx = torch.where(seq < 0)
        node_embeddings[tup_idx] = self.rel_embedding(torch.abs(seq[tup_idx]) - 1)

        return node_embeddings

    def forward(self, batch: Data, tgt_ent_idx=None):
        """
        """
        x = batch['x']

        node_embedding = self.init_seq_embedding(x)
        h, r, ws = self.query_encoder(node_embedding, graph=batch)

        t = self.kge_model(h, r)
        pred = self.kge_model.get_preds(t, tgt_ent_idx)

        return pred, ws

    def pred(self, batch, query_type):
        x = batch['x']
        node_embedding = self.init_seq_embedding(x)

        if query_type in ['1p', '2u-DNF'] and self.use_kge_to_pred_1p:
            # test 1p
            h = node_embedding[:, 2]
            r = node_embedding[:, 3]
            t = self.kge_model(h, r)

            pred = self.kge_model.get_preds(t)

        else:
            h, r, _ = self.query_encoder(node_embedding, graph=batch)
            t = self.kge_model(h, r)
            pred = self.kge_model.get_preds(t)

        return pred

    def train_step(self, optimizer, train_iterator, args, query_type=None):
        """

        :param self:
        :param optimizer:
        :param train_iterator:
        :param args:
        :return:
        """
        self.train()
        optimizer.zero_grad()

        # freeze bn in KGE
        for m in self.kge_model.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()

        t1 = time.time()

        batch = next(train_iterator)

        # t2 = time.time()
        # print('loading ', t2 - t1)
        # t1 = t2

        if args.cuda:
            for k in batch:
                batch[k] = batch[k].cuda()

        with autocast(enabled=self.fp16, dtype=torch.float16):
            negative_samples = batch.pop('negative_samples')
            positive_samples = batch.pop('positive_samples')

            tgt_ent_idx = torch.cat([negative_samples, positive_samples.view(-1, 1)], dim=-1)
            target = torch.LongTensor([tgt_ent_idx.shape[-1] - 1] * tgt_ent_idx.shape[0]).cuda()

            # tgt_ent_idx = None
            # target = positive_samples

            preds, ws = self(batch, tgt_ent_idx=tgt_ent_idx)

            loss, log_preds = self.loss_fct(preds, target)
            neg_log_preds = log_preds[:, :-1].mean().item()
            pos_log_preds = log_preds[:, -1].mean().item()

            loss = loss.mean()

            log = {
                'pos_score': pos_log_preds,
                'neg_score': neg_log_preds,
                'loss': loss.item()
            }

        if self.fp16:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()

        return log

    def test_step(self, easy_answers, hard_answers, args, query_type_to_iterator, query_name_dict,
                  save_result=False, save_str="", save_empty=False, final_test=False):
        self.eval()

        step = 0
        total_steps = sum([len(iterator) for iterator in query_type_to_iterator.values()])
        logs = collections.defaultdict(list)

        with torch.no_grad():
            for query_type, iterator in query_type_to_iterator.items():
                for queries_unflatten, query_types, batch in tqdm(iterator):
                    if args.cuda:
                        for k in batch:
                            batch[k] = batch[k].cuda()

                    if query_types[0] in ['2u-DNF', 'up-DNF']:
                        # [batch, 2, len]
                        origin_x = batch['x']
                        all_preds = []
                        for i in range(2):
                            batch['x'] = origin_x[:, i]
                            pred = self.pred(batch, query_type)
                            all_preds.append(pred)

                        pred = torch.stack(all_preds, dim=1).max(dim=1)[0]
                    else:
                        pred = self.pred(batch, query_type)

                    tmp_logs, tmp_records = calc_metric_and_record(args, easy_answers, hard_answers, pred,
                                                                   queries_unflatten, query_types)
                    for query_structure, res in tmp_logs.items():
                        logs[query_structure].extend(res)

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]]) / len(
                    logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        return metrics


class TokenEmbedding(nn.Module):
    def __init__(self, kwargs):
        super(TokenEmbedding, self).__init__()
        self.kwargs = kwargs
        # [1,2,3]
        if len(kwargs['token_embeddings']):
            self.token_embeds = [int(_) for _ in kwargs['token_embeddings'].split('.')]
        else:
            self.token_embeds = []
        self.hidden_size = kwargs['hidden_size']
        self.dim_ent_embedding = kwargs['dim_ent_embedding']
        self.p_dropout = kwargs['hidden_dropout_prob']

        self.type_embeddings = nn.Embedding(2, self.hidden_size) if 1 in self.token_embeds else None
        self.layer_embeddings = nn.Embedding(8, self.hidden_size) if 2 in self.token_embeds else None
        self.op_embeddings = nn.Embedding(2, self.hidden_size) if 3 in self.token_embeds else None
        self.in_embeddings = nn.Embedding(8, self.hidden_size) if 4 in self.token_embeds else None
        self.out_embeddings = nn.Embedding(8, self.hidden_size) if 5 in self.token_embeds else None

        self.proj = nn.Linear(self.dim_ent_embedding, self.hidden_size)

        self.n_neg_proj = 1
        self.neg_proj = nn.ModuleList(
            [MLP(channel_list=[self.hidden_size, self.hidden_size])
             for _ in range(self.n_neg_proj)]
        )

        self.norm = nn.LayerNorm(self.hidden_size, eps=kwargs['layer_norm_eps'])
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, node_embeddings, graph):
        node_embeddings = self.proj(node_embeddings)

        if self.type_embeddings:
            node_embeddings += self.type_embeddings(graph['node_types'])

        if self.layer_embeddings:
            node_embeddings += self.layer_embeddings(graph['layers'])

        if self.op_embeddings:
            node_embeddings += self.op_embeddings(graph['operators'])

        if self.in_embeddings:
            node_embeddings += self.in_embeddings(graph['in_degs'])

        if self.out_embeddings:
            node_embeddings += self.out_embeddings(graph['out_degs'])

        for i in range(self.n_neg_proj):
            idxes = torch.where(graph['negs'] == i + 1)
            node_embeddings[idxes] = self.neg_proj[i](node_embeddings[idxes])

        node_embeddings = self.norm(node_embeddings)
        node_embeddings = self.dropout(node_embeddings)

        return node_embeddings


class BertEncoder(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs

        self.dim_ent_embedding = kwargs['dim_ent_embedding']
        self.dim_rel_embedding = kwargs['dim_rel_embedding']
        self.hidden_size = kwargs['hidden_size']
        self.num_heads = kwargs['num_attention_heads']
        self.head_dim = kwargs['hidden_size'] // self.num_heads
        self.device = torch.device('cuda') if kwargs['cuda'] else torch.device('cpu')

        self.embedding = TokenEmbedding(kwargs)

        config = BertConfig(
            num_hidden_layers=kwargs['num_hidden_layers'],
            hidden_size=kwargs['hidden_size'],
            num_attention_heads=kwargs['num_attention_heads'],
            intermediate_size=kwargs['intermediate_size'],
            hidden_dropout_prob=kwargs['hidden_dropout_prob'],
            attention_probs_dropout_prob=kwargs['hidden_dropout_prob'],
            fp16=kwargs['fp16'],
            enc_dist=(kwargs['enc_dist'])
        )
        self.bert = BertModel(config)

        self.rev_proj1 = nn.Linear(self.hidden_size, self.dim_ent_embedding)
        self.rev_proj2 = nn.Linear(self.hidden_size, self.dim_rel_embedding)

    def forward(self, initial_node_embeddings,
                graph):
        # [b, l, dim]
        node_embeddings = self.embedding(initial_node_embeddings,
                                         graph)
        # node_embeddings = self.norm(node_embeddings)

        batch, length, dim = node_embeddings.shape

        hidden_states = self.bert(
            inputs_embeds=node_embeddings,
            attention_mask=graph['attention_mask'],
            dist_mat=graph['dist_mat'],
            negs=graph['negs']
        ).last_hidden_state

        # batch, hd
        cls1 = hidden_states[:, 0]
        cls2 = hidden_states[:, 1]
        # tgts = hidden_states[torch.where(graph['targets'] == 1)]

        h = self.rev_proj1(cls1)
        r = self.rev_proj2(cls2)

        return h, r, None


class GNNEncoder(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs

        self.dim_ent_embedding = kwargs['dim_ent_embedding']
        self.dim_rel_embedding = kwargs['dim_rel_embedding']
        self.hidden_size = kwargs['hidden_size']
        self.num_heads = kwargs['num_attention_heads']
        self.p_dropout = kwargs['hidden_dropout_prob']
        self.num_layers = kwargs['num_hidden_layers']
        self.intermediate_size = kwargs['intermediate_size']

        self.device = torch.device('cuda') if kwargs['cuda'] else torch.device('cpu')

        self.embedding = TokenEmbedding(kwargs)

        self.layers = ModuleList([
            TransformerConv(kwargs)
            for _ in range(self.num_layers)
        ])

        self.rev_proj1 = nn.Linear(self.hidden_size, self.dim_ent_embedding)
        self.rev_proj2 = nn.Linear(self.hidden_size, self.dim_rel_embedding)

    def forward(self, initial_node_embeddings,
                layers, node_types, target,
                x, edge_index, batch):
        node_embeddings = self.embedding(initial_node_embeddings,
                                         layers, node_types, target)

        layer_to_node_embedding = []
        for i in range(len(self.layers)):
            layer = self.layers[i]
            node_embeddings = layer(node_embeddings, edge_index, batch)
            layer_to_node_embedding.append(node_embeddings)

        cls1 = node_embeddings[torch.where(x == ENT_CLS)]
        cls2 = node_embeddings[torch.where(x == REL_CLS)]

        h = self.rev_proj1(cls1)
        r = self.rev_proj2(cls2)

        return h, r


class KGE(nn.Module):
    def __init__(self, num_ents, num_rels,
                 dim_ent_embedding, dim_rel_embedding,
                 kwargs):
        super().__init__()
        self.ent_embedding = None
        self.rel_embedding = None

        self.dim_rel_embedding = dim_rel_embedding
        self.dim_ent_embedding = dim_ent_embedding
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.kwargs = kwargs
        self.init_size = 1e-3

    def forward(self, lhs, rel):
        raise NotImplemented

    def load_from_ckpt_path(self, ckpt_path):
        raise NotImplemented

    def get_preds(self, pred_embedding, tgt_ent_idx=None):
        raise NotImplemented

    @staticmethod
    def calc_preds(pred_embedding, ent_embedding, tgt_ent_idx=None):
        if tgt_ent_idx is None:
            # [dim, n_ent]
            tgt_ent_embedding = ent_embedding.weight.transpose(0, 1)

            # [n_batch, n_ent]
            scores = pred_embedding @ tgt_ent_embedding

        else:
            # [n_batch, neg, dim]
            tgt_ent_embedding = ent_embedding(tgt_ent_idx)

            # [n_batch, dim, 1]
            pred_embedding = pred_embedding.unsqueeze(-1)

            scores = torch.bmm(tgt_ent_embedding, pred_embedding)
            scores = scores.squeeze(-1)

        return scores


class Complex(KGE):
    def __init__(self, num_ents, num_rels,
                 dim_ent_embedding, dim_rel_embedding,
                 kwargs):
        super().__init__(num_ents, num_rels,
                         dim_ent_embedding, dim_rel_embedding,
                         kwargs)
        # embedding and W
        self.rank = dim_ent_embedding // 2

        self.ent_embedding = nn.Embedding(num_ents, 2 * self.rank)
        self.rel_embedding = nn.Embedding(num_rels, 2 * self.rank)

        self.ent_embedding.weight.data *= self.init_size
        self.rel_embedding.weight.data *= self.init_size

        self.embeddings = [self.ent_embedding, self.rel_embedding]

    def forward_emb(self, lhs, rel, to_score_idx=None):
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        if not to_score_idx:
            to_score = self.embeddings[0].weight
        else:
            to_score = self.embeddings[0](to_score_idx)

        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        return ((lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
                (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1))

    def forward(self, lhs, rel):
        lhs = torch.chunk(lhs, 2, -1)
        rel = torch.chunk(rel, 2, -1)

        output = ([lhs[0] * rel[0] - lhs[1] * rel[1], lhs[0] * rel[1] + lhs[1] * rel[0]])
        output = torch.cat(output, dim=-1)

        return output

    def get_preds(self, pred_embedding, tgt_ent_idx=None):
        return KGE.calc_preds(pred_embedding, self.ent_embedding, tgt_ent_idx)

    def get_factor(self, x):
        lhs = self.ent_embedding(x[0])
        rel = self.rel_embedding(x[1])
        rhs = self.ent_embedding(x[2])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        return (torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2))

    def load_from_ckpt_path(self, ckpt_path):
        params = torch.load(ckpt_path)
        logging.info(f'loading Complex params from {ckpt_path}')

        try:
            self.embeddings[0].weight.data = params['embeddings.0.weight']
            self.embeddings[1].weight.data = params['embeddings.1.weight']
        except:
            self.embeddings[0].weight.data = params['_entity_embedding.weight']
            self.embeddings[1].weight.data = params['_relation_embedding.weight']

        self.ent_embedding_norm_mean = self.embeddings[0].weight.data.norm(p=2, dim=1).mean().item()
        self.rel_embedding_norm_mean = self.embeddings[1].weight.data.norm(p=2, dim=1).mean().item()

        self.embeddings[0].weight.data /= self.ent_embedding_norm_mean
        self.embeddings[1].weight.data /= self.rel_embedding_norm_mean


class TuckER(KGE):
    def __init__(self, num_ents, num_rels,
                 dim_ent_embedding, dim_rel_embedding,
                 kwargs):
        super().__init__(num_ents, num_rels,
                         dim_ent_embedding, dim_rel_embedding,
                         kwargs)

        self.E = torch.nn.Embedding(num_ents, dim_ent_embedding)
        self.R = torch.nn.Embedding(num_rels, dim_rel_embedding)
        self.W = torch.nn.Parameter(
            torch.tensor(np.random.uniform(-1, 1, (dim_rel_embedding, dim_ent_embedding, dim_ent_embedding)),
                         dtype=torch.float))

        self.input_dropout = torch.nn.Dropout(0.3)
        self.hidden_dropout1 = torch.nn.Dropout(0.4)
        self.hidden_dropout2 = torch.nn.Dropout(0.5)

        self.bn0 = torch.nn.BatchNorm1d(dim_ent_embedding)
        self.bn1 = torch.nn.BatchNorm1d(dim_ent_embedding)

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, lhs, rel):
        x = self.bn0(lhs)
        x = self.input_dropout(x)
        x = x.view(-1, 1, self.dim_ent_embedding)

        W_mat = torch.mm(rel, self.W.view(self.dim_rel_embedding, -1))
        W_mat = W_mat.view(-1, self.dim_ent_embedding, self.dim_ent_embedding)
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, self.dim_ent_embedding)
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        return x

    def get_preds(self, pred_embedding, tgt_ent_idx=None):
        return KGE.calc_preds(pred_embedding, self.E, tgt_ent_idx)

    def load_from_ckpt_path(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path))

        self.ent_embedding = self.E
        self.rel_embedding = self.R


class CP(KGE):
    def __init__(self, num_ents, num_rels,
                 dim_ent_embedding, dim_rel_embedding,
                 kwargs):
        super().__init__(num_ents, num_rels,
                         dim_ent_embedding, dim_rel_embedding,
                         kwargs)

        self.ent_embedding = nn.Embedding(num_ents, self.dim_ent_embedding)
        self.rel_embedding = nn.Embedding(num_rels, self.dim_rel_embedding)
        self.ent_embedding1 = nn.Embedding(num_ents, self.dim_ent_embedding)

        self.ent_embedding.weight.data *= self.init_size
        self.rel_embedding.weight.data *= self.init_size
        self.ent_embedding1.weight.data *= self.init_size

    def forward(self, lhs, rel):
        return lhs * rel

    def get_preds(self, pred_embedding, tgt_ent_idx=None):
        return KGE.calc_preds(pred_embedding, self.ent_embedding1, tgt_ent_idx)

    def load_from_ckpt_path(self, ckpt_path):
        params = torch.load(ckpt_path)
        logging.info(f'loading CP params from {ckpt_path}')

        self.ent_embedding.weight.data = params['lhs.weight']
        self.rel_embedding.weight.data = params['rel.weight']
        self.ent_embedding1.weight.data = params['rhs.weight']


class DistMult(KGE):
    def __init__(self, num_ents, num_rels,
                 dim_ent_embedding, dim_rel_embedding,
                 kwargs):
        super().__init__(num_ents, num_rels,
                         dim_ent_embedding, dim_rel_embedding,
                         kwargs)

        self.ent_embedding = nn.Embedding(num_ents, self.dim_ent_embedding)
        self.rel_embedding = nn.Embedding(num_rels, self.dim_rel_embedding)

        self.ent_embedding.weight.data *= self.init_size
        self.rel_embedding.weight.data *= self.init_size

    def forward(self, lhs, rel):
        return lhs * rel

    def get_preds(self, pred_embedding, tgt_ent_idx=None):
        return KGE.calc_preds(pred_embedding, self.ent_embedding, tgt_ent_idx)

    def load_from_ckpt_path(self, ckpt_path):
        params = torch.load(ckpt_path)
        logging.info(f'loading DistMult params from {ckpt_path}')

        self.ent_embedding.weight.data = params['entity.weight']
        self.rel_embedding.weight.data = params['relation.weight']


class RESCAL(KGE):
    def __init__(self, num_ents, num_rels, dim_ent_embedding, dim_rel_embedding, kwargs):
        super().__init__(num_ents, num_rels,
                         dim_ent_embedding, dim_rel_embedding,
                         kwargs)

        assert dim_rel_embedding == dim_ent_embedding
        self.rank = dim_ent_embedding

        self.ent_embedding = nn.Embedding(num_ents, self.rank)
        self.rel_embedding = nn.Embedding(num_rels, self.rank * self.rank)
        self.ent_embedding.weight.data *= self.init_size
        self.rel_embedding.weight.data *= self.init_size

    def forward(self, lhs, rel):
        rel = rel.view(-1, self.rank, self.rank)
        lhs_proj = lhs.view(-1, 1, self.rank)
        lhs_proj = torch.bmm(lhs_proj, rel).view(-1, self.rank)
        return lhs_proj

    def get_preds(self, pred_embedding, tgt_ent_idx=None):
        return KGE.calc_preds(pred_embedding, self.ent_embedding, tgt_ent_idx)

    def load_from_ckpt_path(self, ckpt_path):
        params = torch.load(ckpt_path)
        logging.info(f'loading RESCAL params from {ckpt_path}')

        self.ent_embedding.weight.data = params['entity.weight']
        self.rel_embedding.weight.data = params['relation.weight']

# class AdaptiveMLP(nn.Module):
#     def __init__(self, hidden_size, dim, n):
#         super().__init__()
#         self.dim = dim
#         self.hidden_size = hidden_size
#         self.n = n
#         self.base = 100
#
#         self.to_w = nn.Linear(self.hidden_size, self.n)
#         self.dW1 = nn.Parameter(torch.randn((self.n, self.hidden_size, self.base)))
#         self.dW2 = nn.Parameter(torch.randn((self.n, self.base, self.dim)))
#
#         self.W = nn.Parameter(torch.randn((self.hidden_size, self.dim)))
#         self.b = nn.Parameter(torch.zeros((self.dim,)))
#
#         self.a = nn.Parameter(torch.tensor([0.1]))
#
#         self.init_weight()
#
#     def init_weight(self):
#         self.dW1.data.normal_(mean=0.0, std=0.02)
#         self.dW2.data.normal_(mean=0.0, std=0.02)
#         self.W.data.normal_(mean=0.0, std=0.02)
#
#     def forward(self, control_msg: Tensor, inputs: Tensor):
#         # [batch, n]
#         w = F.softmax(self.to_w(control_msg), dim=-1)
#
#         # [n, hidden_size, dim]
#         dW = torch.bmm(self.dW1, self.dW2)
#         # weights = self.W
#
#         # [batch, 1, n] [n, hidden_size * dim] -> [batch, hidden_size, dim]
#         dW = torch.matmul(w.unsqueeze(1), dW.view(self.n, -1)).view(-1, self.hidden_size, self.dim)
#
#         weight = self.a * self.W.view(-1, self.hidden_size, self.dim) + (1 - self.a) * dW
#         outputs = torch.bmm(inputs.unsqueeze(1), weight).squeeze() + self.b
#
#         return outputs, None
