import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import collections
from tqdm import tqdm

from Models.base_model import calc_loss
from utils import calc_metric_and_record
from utils.util import DEBUG, get_top1


def Ident(x):
    return x


class BoxOffsetIntersection(nn.Module):

    def __init__(self, dim):
        super(BoxOffsetIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))
        layer1_mean = torch.mean(layer1_act, dim=0)
        gate = torch.sigmoid(self.layer2(layer1_mean))
        offset, _ = torch.min(embeddings, dim=0)

        return offset * gate


class CenterIntersection(nn.Module):

    def __init__(self, dim):
        super(CenterIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)


        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))  # (num_conj, dim)
        attention = F.softmax(self.layer2(layer1_act), dim=0)  # (num_conj, dim)
        embedding = torch.sum(attention * embeddings, dim=0)

        return embedding


class BetaIntersection(nn.Module):

    def __init__(self, dim):
        super(BetaIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(2 * self.dim, 2 * self.dim)
        self.layer2 = nn.Linear(2 * self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, alpha_embeddings, beta_embeddings):
        all_embeddings = torch.cat([alpha_embeddings, beta_embeddings], dim=-1)
        layer1_act = F.relu(self.layer1(all_embeddings))  # (num_conj, batch_size, 2 * dim)
        attention = F.softmax(self.layer2(layer1_act), dim=0)  # (num_conj, batch_size, dim)

        alpha_embedding = torch.sum(attention * alpha_embeddings, dim=0)
        beta_embedding = torch.sum(attention * beta_embeddings, dim=0)

        return alpha_embedding, beta_embedding


class BetaProjection(nn.Module):
    def __init__(self, ent_dim, rel_dim, hidden_dim, projection_regularizer, num_layers):
        super(BetaProjection, self).__init__()
        self.ent_dim = ent_dim
        self.rel_dim = rel_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(self.ent_dim + self.rel_dim, self.hidden_dim)  # 1st layer
        self.layer0 = nn.Linear(self.hidden_dim, self.ent_dim)  # final layer
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)
        self.projection_regularizer = projection_regularizer

    def forward(self, e_embedding, r_embedding):
        x = torch.cat([e_embedding, r_embedding], dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)
        x = self.projection_regularizer(x)

        return x


class Regularizer():
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, ent_embedding):
        return torch.clamp(ent_embedding + self.base_add, self.min_val, self.max_val)


class KGReasoning(nn.Module):
    def __init__(self, num_ent, num_rel, hidden_dim, gamma,
                 geo, test_batch_size=1,
                 box_mode=None, use_cuda=False,
                 query_name_dict=None, beta_mode=None):
        super(KGReasoning, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.geo = geo
        self.use_cuda = use_cuda
        self.batch_ent_range = torch.arange(num_ent).to(torch.float).repeat(test_batch_size,
                                                                                  1).cuda() if self.use_cuda else torch.arange(
            num_ent).to(torch.float).repeat(test_batch_size, 1)  # used in test_step
        self.query_name_dict = query_name_dict

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.ent_dim = hidden_dim
        self.rel_dim = hidden_dim

        if self.geo == 'box':
            self.ent_embedding = nn.Parameter(torch.zeros(num_ent, self.ent_dim))  # centor for entities
            activation, cen = box_mode
            self.cen = cen  # hyperparameter that balances the in-box distance and the out-box distance
            if activation == 'none':
                self.func = Ident
            elif activation == 'relu':
                self.func = F.relu
            elif activation == 'softplus':
                self.func = F.softplus
        elif self.geo == 'vec':
            self.ent_embedding = nn.Parameter(torch.zeros(num_ent, self.ent_dim))  # center for entities
        elif self.geo == 'beta':
            self.ent_embedding = nn.Parameter(torch.zeros(num_ent, self.ent_dim * 2))  # alpha and beta
            self.ent_regularizer = Regularizer(1, 0.05,
                                                  1e9)  # make sure the parameters of beta embeddings are positive
            self.projection_regularizer = Regularizer(1, 0.05,
                                                      1e9)  # make sure the parameters of beta embeddings after rel projection are positive
        nn.init.uniform_(
            tensor=self.ent_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.rel_embedding = nn.Parameter(torch.zeros(num_rel, self.rel_dim))
        nn.init.uniform_(
            tensor=self.rel_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if self.geo == 'box':
            self.offset_embedding = nn.Parameter(torch.zeros(num_rel, self.ent_dim))
            nn.init.uniform_(
                tensor=self.offset_embedding,
                a=0.,
                b=self.embedding_range.item()
            )
            self.center_net = CenterIntersection(self.ent_dim)
            self.offset_net = BoxOffsetIntersection(self.ent_dim)
        elif self.geo == 'vec':
            self.center_net = CenterIntersection(self.ent_dim)
        elif self.geo == 'beta':
            hidden_dim, num_layers = beta_mode
            self.center_net = BetaIntersection(self.ent_dim)
            self.projection_net = BetaProjection(self.ent_dim * 2,
                                                 self.rel_dim,
                                                 hidden_dim,
                                                 self.projection_regularizer,
                                                 num_layers)

    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        if self.geo == 'box':
            return self.forward_box(positive_sample, negative_sample, subsampling_weight, batch_queries_dict,
                                    batch_idxs_dict)
        elif self.geo == 'vec':
            return self.forward_vec(positive_sample, negative_sample, subsampling_weight, batch_queries_dict,
                                    batch_idxs_dict)
        elif self.geo == 'beta':
            return self.forward_beta(positive_sample, negative_sample, subsampling_weight, batch_queries_dict,
                                     batch_idxs_dict)

    def embed_query_box(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using Query2box
        queries: a flattened batch of queries
        '''
        all_rel_flag = True
        for ele in query_structure[
            -1]:  # whether the current query tree has merged to one branch and only need to do rel traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_rel_flag = False
                break
        if all_rel_flag:
            if query_structure[0] == 'e':
                embedding = torch.index_select(self.ent_embedding, dim=0, index=queries[:, idx])
                if self.use_cuda:
                    offset_embedding = torch.zeros_like(embedding).cuda()
                else:
                    offset_embedding = torch.zeros_like(embedding)
                idx += 1
            else:
                embedding, offset_embedding, idx = self.embed_query_box(queries, query_structure[0], idx)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert False, "box cannot handle queries with negation"
                else:
                    r_embedding = torch.index_select(self.rel_embedding, dim=0, index=queries[:, idx])
                    r_offset_embedding = torch.index_select(self.offset_embedding, dim=0, index=queries[:, idx])
                    embedding += r_embedding
                    offset_embedding += self.func(r_offset_embedding)
                idx += 1
        else:
            embedding_list = []
            offset_embedding_list = []
            for i in range(len(query_structure)):
                embedding, offset_embedding, idx = self.embed_query_box(queries, query_structure[i], idx)
                embedding_list.append(embedding)
                offset_embedding_list.append(offset_embedding)
            embedding = self.center_net(torch.stack(embedding_list))
            offset_embedding = self.offset_net(torch.stack(offset_embedding_list))

        return embedding, offset_embedding, idx

    def embed_query_vec(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using GQE
        queries: a flattened batch of queries
        '''
        all_rel_flag = True
        for ele in query_structure[
            -1]:  # whether the current query tree has merged to one branch and only need to do rel traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_rel_flag = False
                break
        if all_rel_flag:
            if query_structure[0] == 'e':
                embedding = torch.index_select(self.ent_embedding, dim=0, index=queries[:, idx])
                idx += 1
            else:
                embedding, idx = self.embed_query_vec(queries, query_structure[0], idx)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert False, "vec cannot handle queries with negation"
                else:
                    r_embedding = torch.index_select(self.rel_embedding, dim=0, index=queries[:, idx])
                    embedding += r_embedding
                idx += 1

                if DEBUG:
                    print(torch.norm(embedding))
                    top10 = get_top1(embedding, self.ent_embedding)
                    print('\n', top10)

        else:
            embedding_list = []
            for i in range(len(query_structure)):
                embedding, idx = self.embed_query_vec(queries, query_structure[i], idx)
                embedding_list.append(embedding)
            embedding = self.center_net(torch.stack(embedding_list))

        return embedding, idx

    def embed_query_beta(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using BetaE
        queries: a flattened batch of queries
        '''
        all_rel_flag = True
        for ele in query_structure[
            -1]:  # whether the current query tree has merged to one branch and only need to do rel traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_rel_flag = False
                break
        if all_rel_flag:
            if query_structure[0] == 'e':
                embedding = self.ent_regularizer(
                    torch.index_select(self.ent_embedding, dim=0, index=queries[:, idx]))
                idx += 1
            else:
                alpha_embedding, beta_embedding, idx = self.embed_query_beta(queries, query_structure[0], idx)
                embedding = torch.cat([alpha_embedding, beta_embedding], dim=-1)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert (queries[:, idx] == -2).all()
                    embedding = 1. / embedding
                else:
                    r_embedding = torch.index_select(self.rel_embedding, dim=0, index=queries[:, idx])
                    embedding = self.projection_net(embedding, r_embedding)
                idx += 1
            alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1)
        else:
            alpha_embedding_list = []
            beta_embedding_list = []
            for i in range(len(query_structure)):
                alpha_embedding, beta_embedding, idx = self.embed_query_beta(queries, query_structure[i], idx)
                alpha_embedding_list.append(alpha_embedding)
                beta_embedding_list.append(beta_embedding)
            alpha_embedding, beta_embedding = self.center_net(torch.stack(alpha_embedding_list),
                                                              torch.stack(beta_embedding_list))

        return alpha_embedding, beta_embedding, idx

    def cal_logit_beta(self, ent_embedding, query_dist):
        alpha_embedding, beta_embedding = torch.chunk(ent_embedding, 2, dim=-1)
        ent_dist = torch.distributions.beta.Beta(alpha_embedding, beta_embedding)
        logit = self.gamma - torch.norm(torch.distributions.kl.kl_divergence(ent_dist, query_dist), p=1, dim=-1)
        return logit

    def forward_beta(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_idxs, all_alpha_embeddings, all_beta_embeddings = [], [], []
        all_union_idxs, all_union_alpha_embeddings, all_union_beta_embeddings = [], [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure] and 'DNF' in self.query_name_dict[query_structure]:
                alpha_embedding, beta_embedding, _ = \
                    self.embed_query_beta(self.transform_union_query(batch_queries_dict[query_structure],
                                                                     query_structure),
                                          self.transform_union_structure(query_structure),
                                          0)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
                all_union_alpha_embeddings.append(alpha_embedding)
                all_union_beta_embeddings.append(beta_embedding)
            else:
                alpha_embedding, beta_embedding, _ = self.embed_query_beta(batch_queries_dict[query_structure],
                                                                           query_structure,
                                                                           0)
                all_idxs.extend(batch_idxs_dict[query_structure])
                all_alpha_embeddings.append(alpha_embedding)
                all_beta_embeddings.append(beta_embedding)

        if len(all_alpha_embeddings) > 0:
            all_alpha_embeddings = torch.cat(all_alpha_embeddings, dim=0).unsqueeze(1)
            all_beta_embeddings = torch.cat(all_beta_embeddings, dim=0).unsqueeze(1)
            all_dists = torch.distributions.beta.Beta(all_alpha_embeddings, all_beta_embeddings)
        if len(all_union_alpha_embeddings) > 0:
            all_union_alpha_embeddings = torch.cat(all_union_alpha_embeddings, dim=0).unsqueeze(1)
            all_union_beta_embeddings = torch.cat(all_union_beta_embeddings, dim=0).unsqueeze(1)
            all_union_alpha_embeddings = all_union_alpha_embeddings.view(all_union_alpha_embeddings.shape[0] // 2, 2, 1,
                                                                         -1)
            all_union_beta_embeddings = all_union_beta_embeddings.view(all_union_beta_embeddings.shape[0] // 2, 2, 1,
                                                                       -1)
            all_union_dists = torch.distributions.beta.Beta(all_union_alpha_embeddings, all_union_beta_embeddings)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_alpha_embeddings) > 0:
                positive_sample_regular = positive_sample[
                    all_idxs]  # positive samples for non-union queries in this batch
                positive_embedding = self.ent_regularizer(
                    torch.index_select(self.ent_embedding, dim=0, index=positive_sample_regular).unsqueeze(1))
                positive_logit = self.cal_logit_beta(positive_embedding, all_dists)
            else:
                positive_logit = torch.Tensor([]).to(self.ent_embedding.device)

            if len(all_union_alpha_embeddings) > 0:
                positive_sample_union = positive_sample[
                    all_union_idxs]  # positive samples for union queries in this batch
                positive_embedding = self.ent_regularizer(
                    torch.index_select(self.ent_embedding, dim=0, index=positive_sample_union).unsqueeze(
                        1).unsqueeze(1))
                positive_union_logit = self.cal_logit_beta(positive_embedding, all_union_dists)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.ent_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_alpha_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = self.ent_regularizer(
                    torch.index_select(self.ent_embedding, dim=0, index=negative_sample_regular.view(-1)).view(
                        batch_size, negative_size, -1))
                negative_logit = self.cal_logit_beta(negative_embedding, all_dists)
            else:
                negative_logit = torch.Tensor([]).to(self.ent_embedding.device)

            if len(all_union_alpha_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = self.ent_regularizer(
                    torch.index_select(self.ent_embedding, dim=0, index=negative_sample_union.view(-1)).view(
                        batch_size, 1, negative_size, -1))
                negative_union_logit = self.cal_logit_beta(negative_embedding, all_union_dists)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.ent_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs + all_union_idxs, \
               torch.stack(torch.cat([all_alpha_embeddings, all_beta_embeddings], dim=-1),
                           torch.cat([all_union_alpha_embeddings, all_union_beta_embeddings], dim=-1))

    def transform_union_query(self, queries, query_structure):
        '''
        transform 2u queries to two 1p queries
        transform up queries to two 2p queries
        '''
        if self.query_name_dict[query_structure] == '2u-DNF':
            queries = queries[:, :-1]  # remove union -1
        elif self.query_name_dict[query_structure] == 'up-DNF':
            queries = torch.cat([torch.cat([queries[:, :2], queries[:, 5:6]], dim=1),
                                 torch.cat([queries[:, 2:4], queries[:, 5:6]], dim=1)], dim=1)
        queries = torch.reshape(queries, [queries.shape[0] * 2, -1])
        return queries

    def transform_union_structure(self, query_structure):
        if self.query_name_dict[query_structure] == '2u-DNF':
            return ('e', ('r',))
        elif self.query_name_dict[query_structure] == 'up-DNF':
            return ('e', ('r', 'r'))

    def cal_logit_box(self, ent_embedding, query_center_embedding, query_offset_embedding):
        delta = (ent_embedding - query_center_embedding).abs()
        distance_out = F.relu(delta - query_offset_embedding)
        distance_in = torch.min(delta, query_offset_embedding)
        logit = self.gamma - torch.norm(distance_out, p=1, dim=-1) - self.cen * torch.norm(distance_in, p=1, dim=-1)
        return logit

    def forward_box(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_center_embeddings, all_offset_embeddings, all_idxs = [], [], []
        all_union_center_embeddings, all_union_offset_embeddings, all_union_idxs = [], [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure]:
                center_embedding, offset_embedding, _ = \
                    self.embed_query_box(self.transform_union_query(batch_queries_dict[query_structure],
                                                                    query_structure),
                                         self.transform_union_structure(query_structure),
                                         0)
                all_union_center_embeddings.append(center_embedding)
                all_union_offset_embeddings.append(offset_embedding)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
            else:
                center_embedding, offset_embedding, _ = self.embed_query_box(batch_queries_dict[query_structure],
                                                                             query_structure,
                                                                             0)
                all_center_embeddings.append(center_embedding)
                all_offset_embeddings.append(offset_embedding)
                all_idxs.extend(batch_idxs_dict[query_structure])

        if len(all_center_embeddings) > 0 and len(all_offset_embeddings) > 0:
            all_center_embeddings = torch.cat(all_center_embeddings, dim=0).unsqueeze(1)
            all_offset_embeddings = torch.cat(all_offset_embeddings, dim=0).unsqueeze(1)
        if len(all_union_center_embeddings) > 0 and len(all_union_offset_embeddings) > 0:
            all_union_center_embeddings = torch.cat(all_union_center_embeddings, dim=0).unsqueeze(1)
            all_union_offset_embeddings = torch.cat(all_union_offset_embeddings, dim=0).unsqueeze(1)
            all_union_center_embeddings = all_union_center_embeddings.view(all_union_center_embeddings.shape[0] // 2, 2,
                                                                           1, -1)
            all_union_offset_embeddings = all_union_offset_embeddings.view(all_union_offset_embeddings.shape[0] // 2, 2,
                                                                           1, -1)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_center_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = torch.index_select(self.ent_embedding, dim=0,
                                                        index=positive_sample_regular).unsqueeze(1)
                positive_logit = self.cal_logit_box(positive_embedding, all_center_embeddings, all_offset_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(self.ent_embedding.device)

            if len(all_union_center_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = torch.index_select(self.ent_embedding, dim=0,
                                                        index=positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_union_logit = self.cal_logit_box(positive_embedding, all_union_center_embeddings,
                                                          all_union_offset_embeddings)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.ent_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_center_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(self.ent_embedding, dim=0,
                                                        index=negative_sample_regular.view(-1)).view(batch_size,
                                                                                                     negative_size, -1)
                negative_logit = self.cal_logit_box(negative_embedding, all_center_embeddings, all_offset_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(self.ent_embedding.device)

            if len(all_union_center_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(self.ent_embedding, dim=0,
                                                        index=negative_sample_union.view(-1)).view(batch_size, 1,
                                                                                                   negative_size, -1)
                negative_union_logit = self.cal_logit_box(negative_embedding, all_union_center_embeddings,
                                                          all_union_offset_embeddings)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.ent_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs + all_union_idxs, \
               torch.stack(all_center_embeddings, all_union_center_embeddings)

    def cal_logit_vec(self, ent_embedding, query_embedding):
        distance = ent_embedding - query_embedding
        logit = self.gamma - torch.norm(distance, p=1, dim=-1)
        return logit

    def forward_vec(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        predict_embeddings = [None] * len(negative_sample)

        all_center_embeddings, all_idxs = [], []
        all_union_center_embeddings, all_union_idxs = [], []

        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure]:
                center_embedding, _ = self.embed_query_vec(
                    self.transform_union_query(batch_queries_dict[query_structure],
                                               query_structure),
                    self.transform_union_structure(query_structure), 0)
                all_union_center_embeddings.append(center_embedding)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
            else:
                center_embedding, _ = self.embed_query_vec(batch_queries_dict[query_structure], query_structure, 0)
                all_center_embeddings.append(center_embedding)
                all_idxs.extend(batch_idxs_dict[query_structure])

        if len(all_center_embeddings) > 0:
            all_center_embeddings = torch.cat(all_center_embeddings, dim=0).unsqueeze(1)

            tmp = all_center_embeddings.squeeze(1)
            for i in range(len(all_idxs)):
                predict_embeddings[all_idxs[i]] = tmp[i]

        if len(all_union_center_embeddings) > 0:
            all_union_center_embeddings = torch.cat(all_union_center_embeddings, dim=0).unsqueeze(1)
            all_union_center_embeddings = all_union_center_embeddings.view(all_union_center_embeddings.shape[0] // 2, 2,
                                                                           1, -1)

            tmp = all_union_center_embeddings.squeeze(2)
            for i in range(len(all_union_idxs)):
                predict_embeddings[all_union_idxs[i]] = tmp[i]

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_center_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = torch.index_select(self.ent_embedding, dim=0,
                                                        index=positive_sample_regular).unsqueeze(1)
                positive_logit = self.cal_logit_vec(positive_embedding, all_center_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(self.ent_embedding.device)

            if len(all_union_center_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = torch.index_select(self.ent_embedding, dim=0,
                                                        index=positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_union_logit = self.cal_logit_vec(positive_embedding, all_union_center_embeddings)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.ent_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_center_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(self.ent_embedding, dim=0,
                                                        index=negative_sample_regular.view(-1)).view(batch_size,
                                                                                                     negative_size, -1)
                negative_logit = self.cal_logit_vec(negative_embedding, all_center_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(self.ent_embedding.device)

            if len(all_union_center_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(self.ent_embedding, dim=0,
                                                        index=negative_sample_union.view(-1)).view(batch_size, 1,
                                                                                                   negative_size, -1)
                negative_union_logit = self.cal_logit_vec(negative_embedding, all_union_center_embeddings)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.ent_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs + all_union_idxs, \
               predict_embeddings

    @staticmethod
    def train_step(self, optimizer, train_iterator, args, query_type):
        self.train()
        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures = next(train_iterator)
        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)
        for i, query in enumerate(batch_queries):  # group queries with same structure
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)
        for query_structure in batch_queries_dict:
            if args.cuda:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
            else:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        positive_logit, negative_logit, subsampling_weight, _, _ = self(positive_sample, negative_sample,
                                                                        subsampling_weight, batch_queries_dict,
                                                                        batch_idxs_dict)

        log, loss = calc_loss(negative_logit, positive_logit, subsampling_weight)
        loss.backward()
        optimizer.step()

        return log

    @staticmethod
    def test_step(model, easy_answers, hard_answers, args, test_dataloader, query_name_dict, save_result=False,
                  save_str="", save_empty=False):
        model.eval()

        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)
        records = collections.defaultdict(dict)

        with torch.no_grad():
            for negative_sample, queries, queries_unflatten, query_structures in tqdm(test_dataloader):
                batch_queries_dict = collections.defaultdict(list)
                batch_idxs_dict = collections.defaultdict(list)
                for i, query in enumerate(queries):
                    batch_queries_dict[query_structures[i]].append(query)
                    batch_idxs_dict[query_structures[i]].append(i)
                for query_structure in batch_queries_dict:
                    if args.cuda:
                        batch_queries_dict[query_structure] = torch.LongTensor(
                            batch_queries_dict[query_structure]).cuda()
                    else:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
                if args.cuda:
                    negative_sample = negative_sample.cuda()

                positive_logit, negative_logit, _, idxs, predict_embeddings = model(None,
                                                                                    negative_sample,
                                                                                    None,
                                                                                    batch_queries_dict,
                                                                                    batch_idxs_dict)
                tmp_logs, tmp_records = calc_metric_and_record(args, easy_answers, hard_answers, negative_logit,
                                                               queries_unflatten,
                                                               query_structures,
                                                               predict_embeddings,
                                                               idxs)
                for query_structure, res in tmp_logs.items():
                    logs[query_structure].extend(res)
                records.update(tmp_records)

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

        # records_path = os.path.join(args.ckpt_path, 'records')
        #
        # if not DEBUG:
        #     logging.info(f'writing predicting records to {records_path}...')
        #     with open(records_path, 'wb') as f:
        #         records_with_meta_data = {
        #             'records': records,
        #             'data_path': args.data_path,
        #             'tasks': args.tasks
        #         }
        #         pickle.dump(records_with_meta_data, f)

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]]) / len(
                    logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        return metrics


