"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn
from numpy.core.multiarray import ndarray
from torch import Tensor

from Graph3dConv import Graph3dConv
from graph_embedding import get_emb
from onmt.GCN import *
from onmt.MYGCN import *
import time
import onmt
from onmt.MPNNs.MPNN import *
import onmt.myutils as myutils
from onmt.encoders.encoder import EncoderBase
# from onmt.utils.misc import aeq
from onmt.modules.position_ffn import PositionwiseFeedForward
# from onmt.encoders import myutils
# from onmt.encoders.MPNN.MPNN import MPNN
from onmt.GATGATE import *
from onmt.modules.util_class import make_condtion


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = onmt.modules.MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = onmt.modules.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Transformer Encoder Layer definition.

        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`

        Returns:
            (`FloatTensor`):

            * outputs `[batch_size x src_len x model_dim]`
        """
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerEncoder(EncoderBase):
    """
    The Transformer encoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings

    Returns:
        (`FloatTensor`, `FloatTensor`):

        * embeddings `[src_len x batch_size x model_dim]`
        * memory_bank `[src_len x batch_size x model_dim]`
    """

    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings, condition_dim, arch):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings
        assert condition_dim >= 0
        self.condition_dim = condition_dim
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])
        if arch in ['transformer', 'after_encoding']:
            d_model += condition_dim
        self.layer_norm = onmt.modules.LayerNorm(d_model)

        self.gcn = Graph3dConv(3, 160, 256, label_dim=21)
        if arch == 'before_linear':
            self.fc = nn.Linear(512+self.condition_dim, 256)
        if arch == 'transformer':
            self.fc = nn.Linear(256, 256)
        else:
            self.fc = nn.Linear(512, 256)
        self.arch = arch

    def forward(self, src, lengths=None):
        """ See :obj:`EncoderBase.forward()`"""
        gs = src[1]
        condition = src[2]
        src = src[0]

        emb = self.embeddings(src)

        out = emb.transpose(0, 1).contiguous()

        cat_list = [out]
        # transformer 模式不加入gcn 模块的编码
        if self.arch != 'transformer':
            emb2 = myutils.gcn_emb(self.gcn, gs, src.device)
            emb2 = emb2.view(src.size(1), -1, 256)
            cat_list.append(emb2)

        if self.condition_dim > 0 and self.arch == 'before_linear':
            condition_of_every_atom = make_condtion(condition, src.size(0), out.device, out.dtype)
            cat_list.append(condition_of_every_atom)

        if len(cat_list) > 1:
            out = torch.cat(cat_list, dim=2)
        else:
            out = cat_list[0]
        out = self.fc(out)

        words = src[:, :, 0].transpose(0, 1)
        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1) \
            .expand(w_batch, w_len, w_len)
        # Run the forward pass of every layer of the tranformer.
        for i in range(self.num_layers):
            out = self.transformer[i](out, mask)

        if self.arch in ['transformer', 'after_encoding']:
            condition_of_every_atom = make_condtion(condition, src.size(0), out.device, out.dtype)
            out = torch.cat((out, condition_of_every_atom), dim=2)
            emb_cond = condition_of_every_atom.transpose(0, 1).contiguous()
            emb = torch.cat((emb, emb_cond), dim=2)

        out = self.layer_norm(out)

        return emb, out.transpose(0, 1).contiguous(), lengths
