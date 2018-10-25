# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#
from time import time
import math
# from profilehooks import profile
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from fairseq.models.transformer_modules import PosEncoding, LayerNormalization
# XD
from fairseq.models.fconv import make_positions, LSTMEncoder
from fairseq.data import LanguagePairDataset


g_init_fn = None


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = Linear(in_features, out_features, bias=bias)
        # init.xavier_normal(self.linear.weight)
        g_init_fn(self.linear.weight)

    def forward(self, inputs):
        return self.linear(inputs)


class Embedding(nn.Module):
    def __init__(self, vocab, d_model, padding_idx=None):
        super(Embedding, self).__init__()
        self.m = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        self.d_model = d_model
        # init.xavier_uniform(self.m.weight)
        self.m.weight.data.normal_(0, self.d_model**-0.5)

    def forward(self, x):
        return self.m(x) * math.sqrt(self.d_model)


# Adapted from OpenNMT-py by XD
class MultiHeadedAttention(nn.Module):
    def __init__(self, k_dim, v_dim, model_dim, head_count, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        assert k_dim == self.dim_per_head
        assert v_dim == self.dim_per_head
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        # Linear without bias, adapted from T2T by XD
        self.linear_keys = Linear(model_dim, head_count * self.dim_per_head, bias=False)
        self.linear_values = Linear(model_dim, head_count * self.dim_per_head, bias=False)
        self.linear_query = Linear(model_dim, head_count * self.dim_per_head, bias=False)
        self.sm = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = Linear(model_dim, model_dim, bias=False)

    def forward(self, query, key, value, attn_mask=None):
        residual = query  # XD

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

        # query = self.layer_norm1(query)
        # key = self.layer_norm2(key)
        # value = self.layer_norm3(value)

        # 1) Project key, value, and query.
        key_up = shape(self.linear_keys(key))       # (batch_size, head_count, k_len, dim_per_head)
        value_up = shape(self.linear_values(value)) # (batch_size, head_count, k_len, dim_per_head)
        query_up = shape(self.linear_query(query))  # (batch_size, head_count, q_len, dim_per_head)

        # 2) Calculate and scale scores.
        query_up = query_up / math.sqrt(dim_per_head)
        scores = torch.matmul(query_up, key_up.transpose(2, 3))  # (batch_size, head_count, q_len, k_len)

        mask = attn_mask
        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(Variable(mask), -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.sm(scores)
        drop_attn = self.dropout(attn)
        context = unshape(torch.matmul(drop_attn, value_up))
        # (batch_size, head_count, q_len, dim_per_head) unshaped to (batch_size, q_len, head_count * dim_per_head)

        output = self.final_linear(context)

        # Return one attn
        top_attn = attn \
            .view(batch_size, head_count,
                  query_len, key_len)[:, 0, :, :] \
            .contiguous()
        return output, top_attn


class PoswiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, relu_dropout=0.1, residual=True):
        super(PoswiseFeedForward, self).__init__()
        # self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        # self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.conv1 = Linear(d_model, d_ff)  # XD
        self.conv2 = Linear(d_ff, d_model)  # XD
        self.layer_norm = LayerNormalization(d_model)

        # Save a little memory, by doing inplace. Copied from OpenNMT-py by XD
        self.relu_dropout = nn.Dropout(relu_dropout, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        global g_init_fn
        if g_init_fn == init.xavier_normal:
            init.kaiming_normal(self.conv1.linear.weight)
        elif g_init_fn == init.xavier_uniform:
            init.kaiming_uniform(self.conv1.linear.weight)

        self.conv1.linear.bias.data.zero_()  # XD
        self.conv2.linear.bias.data.zero_()  # XD
        self.residual = residual

    def forward(self, inputs):
        residual = inputs # inputs: [b_size x len_q x d_model]
        inputs = self.layer_norm(inputs)
        # outputs = self.relu(self.conv1(inputs.transpose(1, 2)))
        # outputs = self.conv2(outputs).transpose(1, 2) # outputs: [b_size x len_q x d_model]
        # # outputs = self.dropout(outputs)
        # outputs = F.dropout(outputs, p = self.dropout, training=self.training)

        # XD
        outputs = self.relu(self.conv1(inputs))
        outputs = self.relu_dropout(outputs)  # added by XD
        outputs = self.conv2(outputs)
        outputs = self.dropout(outputs)

        return residual + outputs if self.residual else outputs


class EncoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, dropout=0.1, args=None):
        super(EncoderLayer, self).__init__()
        self.args = args
        self.enc_self_attn = MultiHeadedAttention(
            d_k, d_v, d_model, n_heads, dropout=args.attention_dropout)
        self.pos_ffn = PoswiseFeedForward(d_model, d_ff, dropout=dropout, relu_dropout=args.relu_dropout)
        self.layer_norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_inputs, self_attn_mask):
        pos_emb = None
        if type(enc_inputs) in [list, tuple]:
            enc_inputs, pos_emb = enc_inputs

        residual = enc_inputs
        enc_inputs = self.layer_norm(enc_inputs)

        if pos_emb is not None: enc_inputs += pos_emb

        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs,
                                               enc_inputs, attn_mask=self_attn_mask)
        enc_outputs = self.dropout(enc_outputs) + residual
        enc_outputs = self.pos_ffn(enc_outputs)

        if pos_emb is not None:
            enc_outputs = (enc_outputs, pos_emb)
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, dropout=0.1, args=None):
        super(DecoderLayer, self).__init__()
        self.args = args
        self.dec_self_attn = MultiHeadedAttention(d_k, d_v, d_model, n_heads, dropout=args.attention_dropout)
        # self.dec_enc_attn = MultiHeadedAttention(d_k, d_v, d_model, n_heads, dropout=dropout)
        self.dec_enc_attn = MultiHeadedAttention(d_k, d_v, d_model, n_heads, dropout=args.attention_dropout)
        self.pos_ffn = PoswiseFeedForward(d_model, d_ff, dropout=dropout, relu_dropout=args.relu_dropout)
        self.layer_norm1 = LayerNormalization(d_model)
        self.layer_norm2 = LayerNormalization(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, dec_inputs, enc_outputs, self_attn_mask, enc_attn_mask):
        residual = dec_inputs
        dec_inputs = self.layer_norm1(dec_inputs)
        dec_outputs, dec_self_attn_score = self.dec_self_attn(dec_inputs, dec_inputs,
                                                        dec_inputs, attn_mask=self_attn_mask)
        dec_outputs = self.dropout1(dec_outputs) + residual
        residual = dec_outputs
        dec_outputs = self.layer_norm2(dec_outputs)
        dec_outputs, dec_enc_attn_score = self.dec_enc_attn(dec_outputs, enc_outputs,
                                                      enc_outputs, attn_mask=enc_attn_mask)
        dec_outputs = self.dropout2(dec_outputs) + residual
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn_score, dec_enc_attn_score


class Encoder(nn.Module):
    """Transformer Encoder """
    def __init__(self, dictionary, n_layers=6, d_k=512, d_v=512, d_model=512, d_ff=512, n_heads=8,
                      max_seq_len=1024, dropout=0.2, weighted=False, args=None):
        super(Encoder, self).__init__()
        self.dictionary = dictionary
        self.src_vocab_size = len(self.dictionary)
        self.padding_idx = self.dictionary.pad()
        self.d_model = d_model
        self.dropout = dropout
        self.src_emb = Embedding(self.src_vocab_size, self.d_model, padding_idx=self.padding_idx)
        self.pos_emb = PosEncoding(max_seq_len, d_model) # minus *10
        self.max_pos = max_seq_len  # XD
        self.dropout_emb = nn.Dropout(dropout) if args.embedding_dropout > 0 else None
        self.hold_pos_enc = args.hold_pos_enc if hasattr(args, 'hold_pos_enc') else False
        self.sep_enc_pos = args.sep_enc_pos if hasattr(args, 'sep_enc_pos') else False
        self.layer_type = EncoderLayer
        self.layer_norm = LayerNormalization(d_model)
        self.layers = nn.ModuleList(
            [self.layer_type(d_k, d_v, d_model, d_ff, n_heads, dropout, args=args) for _ in range(n_layers)])

    def forward(self, tokens, src_features=None, target_id=None, return_attn=False):
        positions = Variable(make_positions(tokens.data, self.dictionary.pad(),
                                            left_pad=LanguagePairDataset.LEFT_PAD_SOURCE))
        # enc_input_len = positions.size(1)
        pos_emb = self.pos_emb(positions)
        enc_outputs = self.src_emb(tokens)
        if self.sep_enc_pos:
            if self.dropout_emb:
                enc_outputs = self.dropout_emb(enc_outputs)
            enc_outputs = (enc_outputs, pos_emb)
        else:
            if self.dropout_emb is not None and self.hold_pos_enc:
                enc_outputs = self.dropout_emb(enc_outputs)
            enc_outputs += pos_emb
            if self.dropout_emb is not None and not self.hold_pos_enc:
                enc_outputs = self.dropout_emb(enc_outputs)

        enc_self_attn_mask = get_attn_pad_mask(tokens, tokens)
        enc_self_attns = []
        for i,layer in enumerate(self.layers):
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            if return_attn:
                enc_self_attns.append(enc_self_attn)
        if self.sep_enc_pos:
            enc_outputs = enc_outputs[0] + pos_emb
        enc_outputs = self.layer_norm(enc_outputs)
        enc_outputs = (enc_outputs, tokens)
        return enc_outputs #, enc_self_attns

    # XD
    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_pos - self.dictionary.pad() - 1

    def upgrade_state_dict(self, state_dict):
        return state_dict


class Decoder(nn.Module):
    """Transformer Decoder"""
    def __init__(self, dictionary, n_layers=6, d_k=512, d_v=512, d_model=512, d_ff=512, n_heads=8,
                      max_seq_len=1024, dropout=0.2, weighted=False, args=None):
        super(Decoder, self).__init__()
        self.dictionary = dictionary
        self.tgt_vocab_size = len(self.dictionary)
        self.padding_idx = self.dictionary.pad()
        self.d_model = d_model
        self.dropout = dropout
        self.tgt_emb = Embedding(self.tgt_vocab_size, self.d_model, padding_idx=self.padding_idx)
            # padding_idx=self.padding_idx if args.zero_padding_embedding else None)
        self.pos_emb = PosEncoding(max_seq_len, d_model) # minus *10
        self.max_pos = max_seq_len  # XD
        # self.pos_emb = nn.Embedding(max_seq_len, self.d_model, padding_idx=self.padding_idx)
        self.dropout_emb = nn.Dropout(dropout) if args.embedding_dropout > 0 else None
        self.hold_pos_enc = args.hold_pos_enc if hasattr(args, 'hold_pos_enc') else False
        self.layer_norm = LayerNormalization(d_model)
        if args.pre_output_proj:
            self.proj = Linear(self.d_model, self.d_model, bias=False)
            self.proj_dropout = nn.Dropout(dropout)
        else:
            self.proj, self.proj_dropout = None, None
        self.separate_pos_enc = args.separate_pos_enc \
            if hasattr(args, 'separate_pos_enc') else False
        self.layer_type = DecoderLayer
        self.layers = nn.ModuleList(
            [self.layer_type(d_k, d_v, d_model, d_ff, n_heads, dropout, args=args) for _ in range(n_layers)])

        # XD
        self.fc = Linear(self.d_model, self.tgt_vocab_size, bias=args.output_bias if hasattr(args, 'output_bias') else True)
        self.fc.weight.data.normal_(0, self.d_model**-0.5)  # same as Embedding
        if not hasattr(args, 'output_bias') or args.output_bias:
            self.fc.bias.data.zero_()
        if args.share_input_output_embedding:
            self.fc.weight = self.tgt_emb.m.weight

    def _split_enc_outputs(self, enc_outputs):
        enc_inputs = enc_outputs[-1]
        enc_outputs = enc_outputs[:-1]
        if len(enc_outputs) == 1:
            enc_outputs = enc_outputs[0]
        else:
            enc_outputs = enc_outputs[0]
        return enc_inputs, enc_outputs

    # @profile
    def forward(self, tokens, enc_outputs, return_attn=False):
        enc_inputs, enc_outputs = self._split_enc_outputs(enc_outputs)
        positions = Variable(make_positions(tokens.data, self.dictionary.pad(),
                                            left_pad=LanguagePairDataset.LEFT_PAD_TARGET))
        # dec_input_len = positions.size(1)
        pos_emb = self.pos_emb(positions)
        dec_outputs = self.tgt_emb(tokens)
        if self.dropout_emb is not None and self.hold_pos_enc:
            dec_outputs = self.dropout_emb(dec_outputs)
        dec_outputs += pos_emb
        if self.dropout_emb is not None and not self.hold_pos_enc:
            dec_outputs = self.dropout_emb(dec_outputs)

        dec_self_attn_pad_mask = get_attn_pad_mask(tokens, tokens)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(tokens, size=self.max_pos+16)

        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_enc_attn_pad_mask = get_attn_pad_mask(tokens, enc_inputs)

        # dec_self_attns, dec_enc_attns = [], []
        avg_attn_scores = None
        for i,layer in enumerate(self.layers):
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs,
                                                             self_attn_mask=dec_self_attn_mask,
                                                             enc_attn_mask=dec_enc_attn_pad_mask)
            attn_scores = dec_enc_attn / len(self.layers)
            if avg_attn_scores is None:
                avg_attn_scores = attn_scores
            else:
                avg_attn_scores.add_(attn_scores)
            # if return_attn:
            #     dec_self_attns.append(dec_self_attn)
            #     dec_enc_attns.append(dec_enc_attn)
        if self.separate_pos_enc:
            dec_outputs = dec_outputs - pos_emb
        dec_outputs = self.layer_norm(dec_outputs)
        if self.proj is not None:
            dec_outputs = self.proj_dropout(self.proj(dec_outputs))
        dec_outputs = self.fc(dec_outputs)
        # return dec_outputs, dec_self_attns, dec_enc_attns
        return dec_outputs, avg_attn_scores

    # XD
    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_pos - self.dictionary.pad() - 1

    def upgrade_state_dict(self, state_dict):
        return state_dict


class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.src_dict = encoder.dictionary
        self.dst_dict = decoder.dictionary

        assert self.src_dict.pad() == self.dst_dict.pad()
        assert self.src_dict.eos() == self.dst_dict.eos()
        assert self.src_dict.unk() == self.dst_dict.unk()

    # XD
    def forward(self, src_tokens, input_tokens, src_features, target_id=None, return_attn=False):
        # encoder_out, encoder_self_attns = self.encoder(src_tokens, src_features=src_features, target_id=target_id)
        # decoder_out, decoder_self_attns, decoder_enc_attns = self.decoder(input_tokens, src_tokens, encoder_out, return_attn=return_attn)
        encoder_out = self.encoder(src_tokens, src_features=src_features, target_id=target_id)
        decoder_out, decoder_enc_attns = self.decoder(input_tokens, encoder_out, return_attn=return_attn)
        return decoder_out.view(-1, decoder_out.size(-1))

    def trainable_parameters(self):
        # Avoid update the position enccoding
        return filter(lambda p: p.requires_grad, self.parameters())

    def max_encoder_positions(self):
        """Maximum input length supported by the encoder."""
        return self.encoder.max_positions()

    def max_decoder_positions(self):
        """Maximum output length supported by the decoder."""
        return self.decoder.max_positions()

    def upgrade_state_dict(self, state_dict):
        state_dict = self.encoder.upgrade_state_dict(state_dict)
        state_dict = self.decoder.upgrade_state_dict(state_dict)
        return state_dict

    def make_generation_fast_(self, **kwargs):
        return


def _get_triu_mask(attn_shape, is_cuda=True):
    # t0 = time()
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if is_cuda:
        subsequent_mask = subsequent_mask.cuda()
    # print('In _get_triu_mask', time() - t0)
    return subsequent_mask


def get_attn_pad_mask(seq_q, seq_k, padding_idx=1):
    b_size, len_q = seq_q.size()
    b_size, len_k = seq_k.size()
    return seq_k.data.eq(padding_idx).unsqueeze(1).expand(b_size, len_q, len_k)


def get_attn_subsequent_mask(seq, size=1024):
    # t0 = time()
    if not hasattr(get_attn_subsequent_mask, 'mask'):
        get_attn_subsequent_mask.mask = _get_triu_mask((1, size, size), is_cuda=seq.is_cuda)
    mask = get_attn_subsequent_mask.mask[:, :seq.size(1), :seq.size(1)]
    # print('In get_attn_subsequent_mask', time() - t0)
    return mask


def get_archs():
    return [
        'transformer_caiyun',
        'transformer_big'
    ]

def _check_arch(args):
    """Check that the specified architecture is valid and not ambiguous."""
    if args.arch not in get_archs():
        raise ValueError('Unknown fconv model architecture: {}'.format(args.arch))
    if args.arch != 'fconv':
        # check that architecture is not ambiguous
        for a in ['encoder_embed_dim', 'encoder_layers', 'decoder_embed_dim', 'decoder_layers',
                  'decoder_out_embed_dim']:
            if hasattr(args, a):
                raise ValueError('--{} cannot be combined with --arch={}'.format(a, args.arch))


def parse_arch(args):
    _check_arch(args)

    if args.arch == 'transformer_caiyun':
        args.n_layers = 6
        args.d_k = 64
        args.d_v = 64
        args.d_model = 512
        args.d_ff = 2048
        args.n_heads = 8
        args.max_seq_len = 1024
        args.dropout = 0.1
        args.attention_dropout = 0.1
        args.relu_dropout = 0.1
        args.weighted = False
        # args.share_input_output_embedding = True
        if args.warmup_updates > 0:
            assert args.adam_betas == '(0.9, 0.999)'
            args.adam_betas = '(0.9, 0.997)'
        if not hasattr(args, 'output_bias'):
            args.output_bias = True
    elif args.arch == 'transformer_big':
        args.n_layers = 6
        args.d_k = 64
        args.d_v = 64
        args.d_model = 1024
        args.d_ff = 4096
        args.n_heads = 16
        args.max_seq_len = 1024
        args.dropout = 0.3
        args.attention_dropout = 0.1
        args.relu_dropout = 0.1
        args.weighted = False
        # args.share_input_output_embedding = True
        if args.warmup_updates > 0:
            assert args.adam_betas == '(0.9, 0.999)'
            args.adam_betas = '(0.9, 0.997)'

    # default architecture
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', '[(512, 3)] * 20')
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', '[(512, 3)] * 20')
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.decoder_attention = getattr(args, 'decoder_attention', 'True')

    return args


def build_model2(args, src_dict, dst_dict):
    for arg_name in ['query_type', 'query_addend']:
        for arg_prefix in ['enc_', 'dec_', 'attn_']:
            new_arg_name = arg_prefix + arg_name
            if not hasattr(args, new_arg_name):
                assert hasattr(args, arg_name)
                print('set', new_arg_name, 'to', arg_name, 'with value', getattr(args, arg_name))
                setattr(args, new_arg_name, getattr(args, arg_name))

    encoder = TransformerEncoder(
        src_dict,
        n_layers=args.n_layers,
        d_k=args.d_k,
        d_v=args.d_v,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=args.n_heads,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        weighted=args.weighted,
        args=args
    )
    decoder = TransformerDecoder(
        dst_dict,
        n_layers=args.n_layers,
        d_k=args.d_k,
        d_v=args.d_v,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=args.n_heads,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        weighted=args.weighted,
        args=args
    )
    return Transformer(encoder, decoder)


def build_model(args, src_dict, dst_dict):
    global g_init_fn
    if args.init_fn == 'xavier_normal':
        g_init_fn = init.xavier_normal
    elif args.init_fn == 'xavier_uniform':
        g_init_fn = init.xavier_uniform

    if args.encoder_type == 'lstm':
        encoder = LSTMEncoder(src_dict, single_output=True)
    else:
        encoder = Encoder(
            src_dict,
            n_layers=args.n_layers,
            d_k=args.d_k,
            d_v=args.d_v,
            d_model=args.d_model,
            d_ff=args.d_ff,
            n_heads=args.n_heads,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
            args=args,
            weighted=args.weighted
        )
    decoder = Decoder(
        dst_dict,
        n_layers=args.n_layers,
        d_k=args.d_k,
        d_v=args.d_v,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=args.n_heads,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        args=args,
        weighted=args.weighted
    )
    return Transformer(encoder, decoder)
