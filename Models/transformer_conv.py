import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GraphNorm, InstanceNorm, BatchNorm
from torch.nn import BatchNorm1d
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing, RGATConv
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import softmax

import torch.nn as nn
from transformers.activations import ACT2FN


class SelfOutput(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.hidden_size = kwargs['hidden_size']
        self.dense = nn.Linear(kwargs['hidden_size'], kwargs['hidden_size'])
        self.norm = nn.LayerNorm(kwargs['hidden_size'], eps=kwargs['layer_norm_eps'])
        self.dropout = nn.Dropout(kwargs['hidden_dropout_prob'])

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, batch) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states += input_tensor

        # # [batch, num_nodes, hidden_size]
        # hidden_states = hidden_states.view(num_graphs, -1, self.hidden_size)
        hidden_states = self.norm(hidden_states)
        # hidden_states = hidden_states.view(-1, self.hidden_size)

        return hidden_states


class Intermediate(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.dense = nn.Linear(kwargs['hidden_size'], kwargs['intermediate_size'])
        self.intermediate_act_fn = ACT2FN['gelu']

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

    def reset_parameters(self):
        self.dense.reset_parameters()


class Output(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.hidden_size = kwargs['hidden_size']
        self.dense = nn.Linear(kwargs['intermediate_size'], kwargs['hidden_size'])
        self.norm = nn.LayerNorm(kwargs['hidden_size'], eps=kwargs['layer_norm_eps'])
        self.dropout = nn.Dropout(kwargs['hidden_dropout_prob'])

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, batch) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states += input_tensor

        # # [batch, num_nodes, hidden_size]
        # hidden_states = hidden_states.view(num_graphs, -1, self.hidden_size)
        hidden_states = self.norm(hidden_states)
        # hidden_states = hidden_states.view(-1, self.hidden_size)

        return hidden_states


class TransformerConv(MessagePassing):
    r"""The graph transformer operator from the `"Masked Label Prediction:
    Unified Message Passing Model for Semi-Supervised Classification"
    <https://arxiv.org/abs/2009.03509>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \mathbf{W}_2 \mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed via
    multi-head dot product attention:

    .. math::
        \alpha_{i,j} = \textrm{softmax} \left(
        \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top} (\mathbf{W}_4\mathbf{x}_j)}
        {\sqrt{d}} \right)

    Args:
        hidden_size (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_size (int): Size of each output sample.
        num_attention_heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        beta (bool, optional): If set, will combine aggregation and
            skip information via

            .. math::
                \mathbf{x}^{\prime}_i = \beta_i \mathbf{W}_1 \mathbf{x}_i +
                (1 - \beta_i) \underbrace{\left(\sum_{j \in \mathcal{N}(i)}
                \alpha_{i,j} \mathbf{W}_2 \vec{x}_j \right)}_{=\mathbf{m}_i}

            with :math:`\beta_i = \textrm{sigmoid}(\mathbf{w}_5^{\top}
            [ \mathbf{W}_1 \mathbf{x}_i, \mathbf{m}_i, \mathbf{W}_1
            \mathbf{x}_i - \mathbf{m}_i ])` (default: :obj:`False`)
        hidden_dropout_prob (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). Edge features are added to the keys after
            linear transformation, that is, prior to computing the
            attention dot product. They are also added to final values
            after the same linear transformation. The model is:

            .. math::
                \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
                \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \left(
                \mathbf{W}_2 \mathbf{x}_{j} + \mathbf{W}_6 \mathbf{e}_{ij}
                \right),

            where the attention coefficients :math:`\alpha_{i,j}` are now
            computed via:

            .. math::
                \alpha_{i,j} = \textrm{softmax} \left(
                \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top}
                (\mathbf{W}_4\mathbf{x}_j + \mathbf{W}_6 \mathbf{e}_{ij})}
                {\sqrt{d}} \right)

            (default :obj:`None`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add the transformed root node features to the output and the
            option  :attr:`beta` is set to :obj:`False`. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(
            self,
            kwargs=None,
            concat: bool = True,
            edge_dim: Optional[int] = None,
            beta: bool = False,
            root_weight: bool = False,
    ):
        kwargs.setdefault('aggr', 'add')
        super(TransformerConv, self).__init__(node_dim=0, **kwargs)

        self.hidden_size = kwargs['hidden_size']
        self.num_attention_heads = kwargs['num_attention_heads']
        self.p_dropout = kwargs['hidden_dropout_prob']

        self.head_dim = self.hidden_size // self.num_attention_heads
        assert self.head_dim * kwargs[
            'num_attention_heads'] == self.hidden_size, "embed_dim must be divisible by num_attention_heads"

        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.edge_dim = edge_dim
        self._alpha = None

        self.lin_query = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim)
        self.lin_key = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim)
        self.lin_value = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim)

        # self.max_depth_embeddings = nn.Embedding(3, self.hidden_size)

        # if self.root_weight:
        #     self.lin_skip = nn.Linear(self.hidden_size, self.hidden_size)
        # else:
        #     self.lin_skip = None

        if self.beta:
            self.lin_beta = nn.Linear(3 * self.hidden_size, 1, bias=False)
        else:
            self.lin_beta = None

        self.self_output = SelfOutput(kwargs)
        self.intermediate = Intermediate(kwargs)
        self.output = Output(kwargs)

    def forward(self, x, edge_index: Adj, batch, node_to_max_depth=None,
                edge_attr: OptTensor = None, return_attention_weights=None):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        H, C = self.num_attention_heads, self.head_dim

        query = self.lin_query(x).view(-1, H, C)
        key = self.lin_key(x).view(-1, H, C)
        value = self.lin_value(x).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        self_output = self.propagate(edge_index, query=query, key=key, value=value,
                                     edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            self_output = self_output.view(-1, self.num_attention_heads * self.head_dim)
        else:
            self_output = self_output.mean(dim=1)

        attention_output = self.self_output(self_output, x, batch)
        intermediate_output = self.intermediate(attention_output)
        output = self.output(intermediate_output, attention_output, batch)

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return self_output, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return self_output, edge_index.set_value(alpha, layout='coo')
        else:
            return output

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        # if self.lin_edge is not None:
        #     assert edge_attr is not None
        #     edge_attr = self.lin_edge(edge_attr).view(-1, self.num_attention_heads,
        #                                               self.hidden_size)
        #     key_j += edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.head_dim)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.p_dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out += edge_attr

        out *= alpha.view(-1, self.num_attention_heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.hidden_size}, '
                f'{self.hidden_size}, heads={self.num_attention_heads})')
