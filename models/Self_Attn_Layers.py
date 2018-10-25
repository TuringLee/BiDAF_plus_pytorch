#!/usr/bin/enc python3
# BiDAF model of Jarvis for SQuAD .
# Wirtten by Turing Lee in ColorfulClouds .

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import math
from .Modules import PosEncoding, Linear, make_positions

RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

class Multi_Head_Self_Attn(nn.Module):
    """
    Add Self Attention on BiDAF.
    """
    def __init__(self, args):
        super(Multi_Head_Self_Attn, self).__init__()
        self.args = args
        self.inp_linear = Linear(args.model_dim * 2, args.model_dim * 6, bias = False)
        self.output_linear = Linear(args.model_dim * 2, args.model_dim * 2, bias = False)
       
        self.conv1 = Linear(args.model_dim * 2, args.model_dim * 8)
        self.conv2 = Linear(args.model_dim * 8, args.model_dim * 2)
        self.layer_norm = nn.LayerNorm(args.model_dim * 2)

        self.conv1.bias.data.zero_()  # XD
        self.conv2.bias.data.zero_()  # XD

    def forward(self, model_output, context_mask):

        self_attn = self.multi_head_attn(model_output, context_mask)
        self_attn = self.feedforwardNet(self_attn)

        return self_attn

    def multi_head_attn(self, model_output, context_mask):
        # residual = model_output
        num_heads = self.args.num_heads
        batch_size, time_steps, inp_dim = model_output.size()

        projection_model_output = self.inp_linear(model_output)
        queries, keys, values = projection_model_output.split(self.args.model_dim * 2, -1)
        context_mask_self_attn = context_mask.unsqueeze(2).expand(context_mask.size(0), context_mask.size(1), self.args.model_dim * 2)

        # Shape to Batch_size*num_head x time_steps x Model_out_dim/num_head
        query_per_head = self.shape_to_per_head(queries, context_mask_self_attn, batch_size, time_steps, inp_dim, num_heads)
        key_per_head = self.shape_to_per_head(keys, context_mask_self_attn, batch_size, time_steps, inp_dim, num_heads)
        value_per_head = self.shape_to_per_head(values, context_mask_self_attn, batch_size, time_steps, inp_dim, num_heads)

        query_per_head = query_per_head / math.sqrt(inp_dim / num_heads)

        # Batch_size * num_head x time_steps x time_steps
        scaled_similarities = torch.matmul(query_per_head, key_per_head.transpose(2, 3))

        context_mask_similarities_1 = context_mask.unsqueeze(2).expand(batch_size, time_steps, time_steps)
        context_mask_similarities_2 = context_mask.unsqueeze(1).expand(batch_size, time_steps, time_steps)
        diag_mask = torch.eye(time_steps)
        diag_mask = diag_mask.unsqueeze(0).expand(batch_size, time_steps, time_steps).type(torch.ByteTensor).cuda()
        context_mask_similarities = torch.ge(context_mask_similarities_1 + context_mask_similarities_2 + diag_mask, 1)
        # context_mask_similarities = context_mask_similarities.repeat(num_heads, 1, 1)
        context_mask_similarities = context_mask_similarities.unsqueeze(1).expand(batch_size, num_heads, time_steps, time_steps)

        scaled_similarities.data.masked_fill_(context_mask_similarities, -1e10)
        # scaled_similarities.data.masked_fill_(diag_mask, -1e10)
        scaled_similarities = F.softmax(scaled_similarities, dim=-1)
        scaled_similarities = F.dropout(scaled_similarities, p = self.args.multi_head_dropout, training = self.training)

        self_attn = torch.matmul(scaled_similarities, value_per_head)
        self_attn = self_attn.transpose(1, 2).contiguous()
        self_attn = self_attn.view(batch_size, time_steps, inp_dim)
        self_attn.data.masked_fill_(context_mask_self_attn, 0)
        self_attn = self.output_linear(self_attn)
        self_attn.data.masked_fill_(context_mask_self_attn, 0)

        return self_attn

    def feedforwardNet(self, inputs):
        residual = inputs
        inputs = self.layer_norm(inputs)
        outputs = F.relu(self.conv1(inputs))
        outputs = F.dropout(outputs, p = self.args.multi_head_dropout, training=self.training)
        outputs = self.conv2(outputs)
        outputs = F.dropout(outputs, p = self.args.multi_head_dropout, training=self.training)

        return residual + outputs

    def shape_to_per_head(self, inp, mask, batch_size, time_steps, inp_dim, num_heads):
        # shape to B x N x T x C
        # N : num_head
        # C : dim_per_head
        if mask is not None:
            inp.data.masked_fill_(mask, 0)
        inp_per_head = inp.view(batch_size, time_steps, num_heads, inp_dim/num_heads)
        inp_per_head = inp_per_head.transpose(1, 2).contiguous()
        
        return inp_per_head


# Adapted from OpenNMT-py by XD
class MultiHeadedAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadedAttention, self).__init__()
        self.args = args

        model_dim = self.args.model_dim * 2
        head_count = self.args.num_heads

        self.dim_per_head = model_dim // head_count

        self.model_dim = model_dim

        self.head_count = head_count

        self.pos_emb = PosEncoding(400 + 100, model_dim)
        self.inp_linear = Linear(model_dim, model_dim * 3, bias = False)
        # self.linear_keys = Linear(model_dim, model_dim, bias=False)
        # self.linear_values = Linear(model_dim, model_dim, bias=False)
        # self.linear_query = Linear(model_dim, model_dim, bias=False)

        self.final_linear = Linear(model_dim, model_dim, bias=False)
        self.multi_head_layer_norm = nn.LayerNorm(model_dim)

        # FFN
        self.conv1 = Linear(model_dim, model_dim * 4)  # ltj
        self.conv2 = Linear(model_dim * 4, model_dim)  # ltj
        self.ffn_layer_norm = nn.LayerNorm(model_dim)
        self.out_ffn_layer_norm = nn.LayerNorm(model_dim)

        init.kaiming_normal(self.conv1.linear.weight)

        self.conv1.linear.bias.data.zero_()  # XD
        self.conv2.linear.bias.data.zero_()  # XD

    def forward(self, inp, attn_mask=None):

        batch_size = inp.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = inp.size(1)
        query_len = inp.size(1)

        positions = make_positions(batch_size, query_len)
        positions = Variable(positions.type(torch.LongTensor)).cuda()
        position_emb = self.pos_emb(positions)

        # inp = inp + position_emb
        # residual_multi_head = inp
        inp = F.dropout(inp, p = self.args.multi_head_dropout, training = self.training)
        inp = self.multi_head_layer_norm(inp) 
        inp = inp + position_emb
        residual_multi_head = inp

        query, key, value = torch.chunk(self.inp_linear(inp), 3, -1)

        def shape(x):
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        key_up = shape(key)       # (batch_size, head_count, k_len, dim_per_head)
        value_up = shape(value) # (batch_size, head_count, k_len, dim_per_head)
        query_up = shape(query)  # (batch_size, head_count, q_len, dim_per_head)

        # key_up = shape(self.linear_keys(inp))       # (batch_size, head_count, k_len, dim_per_head)
        # value_up = shape(self.linear_values(inp)) # (batch_size, head_count, k_len, dim_per_head)
        # query_up = shape(self.linear_query(inp))  # (batch_size, head_count, q_len, dim_per_head)

        # 2) Calculate and scale scores.
        query_up = query_up / math.sqrt(dim_per_head)
        scores = torch.matmul(query_up, key_up.transpose(2, 3))  # (batch_size, head_count, q_len, k_len)

        mask = attn_mask
        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(Variable(mask), -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = F.softmax(scores, dim=-1)
        drop_attn = F.dropout(attn, p = self.args.multi_head_dropout, training = self.training)
        context = unshape(torch.matmul(drop_attn, value_up))
        # (batch_size, head_count, q_len, dim_per_head) unshaped to (batch_size, q_len, head_count * dim_per_head)

        output = self.final_linear(context)
        output = F.dropout(output, p = self.args.multi_head_dropout, training = self.training)
        output = output + residual_multi_head

        # FFN
        # inputs: [b_size x len_q x d_model]
        # residual_ffn = output
        inputs = self.ffn_layer_norm(output)
        residual_ffn = inputs
        # XD
        outputs = F.relu(self.conv1(inputs))
        outputs = F.dropout(outputs, p = self.args.multi_head_dropout, training = self.training)  # added by XD
        outputs = self.conv2(outputs)
        outputs = F.dropout(outputs, p = self.args.multi_head_dropout, training = self.training)
        outputs = self.out_ffn_layer_norm(residual_ffn + outputs) 
        return outputs


class TriLinear_Self_Attn(nn.Module):
    """
    Trilinear self attention.
    Same as https://github.com/allenai/allennlp/pull/346/files
    """
    def __init__(self, args):
        super(TriLinear_Self_Attn, self).__init__()
        self.args = args
        self.inp_dim = self.args.model_dim * 2
        # self.LSTM = RNN_TYPES[args.rnn_type](input_size = args.model_dim * 2, hidden_size = args.model_dim, 
        #                     num_layers = 1, bidirectional = True)
        
        self.query_weights = nn.Parameter(torch.FloatTensor(self.inp_dim, 1))
        self.key_weights = nn.Parameter(torch.FloatTensor(self.inp_dim, 1))
        self.dot_weights = nn.Parameter(torch.FloatTensor(1, 1, self.inp_dim))

        self.linear_inner = Linear(self.inp_dim * 3, self.inp_dim)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform(self.query_weights)
        init.xavier_uniform(self.key_weights)
        init.xavier_uniform(self.dot_weights)
        # for p in self.LSTM.parameters():
        #     p.data.normal_(0, 0.05)
        
    def forward(self, model_output, context_mask):

        residual = model_output

        model_output = F.dropout(model_output, p = self.args.dropout, training = self.training)
        # model_output, _ = self.LSTM(model_output)
        # model_output = F.dropout(model_output, p = self.args.dropout, training = self.training)
        
        query = model_output
        key = model_output
        # value = model_output.clone()

        batch_size, time_steps, inp_dim = query.size()

        # query (B x T x C) * (C x 1) --> (B x T x 1) 
        query_factors = torch.matmul(query.contiguous().view(batch_size*time_steps, inp_dim), self.query_weights)
        query_factors = query_factors.contiguous().view(batch_size, time_steps, 1)
        # key (B x T x C) * (C x 1) --> (B x 1 x T) 
        key_factors = torch.matmul(key.contiguous().view(batch_size*time_steps, inp_dim), self.key_weights)
        key_factors = key_factors.contiguous().view(batch_size, time_steps, 1).transpose(1, 2)

        # query (B x T x C) * (1 x 1 x C) --> (B x T x C)
        query_weighted = query * self.dot_weights

        # (B x T x C) * (B x C x T) --> similar to attn_scores(B x T x T)
        dot_factors = torch.matmul(query_weighted, key.transpose(1, 2))

        attn_scores = dot_factors + query_factors + key_factors

        # Create attn score mask
        attn_scores_mask_1 = context_mask.unsqueeze(2).expand(context_mask.size(0), context_mask.size(1), context_mask.size(1))
        attn_scores_mask_2 = context_mask.unsqueeze(1).expand(context_mask.size(0), context_mask.size(1), context_mask.size(1))
        attn_scores_mask = torch.ge(attn_scores_mask_1 + attn_scores_mask_2 ,1)

        diag_mask = torch.eye(time_steps)
        diag_mask = diag_mask.unsqueeze(0).expand(batch_size, diag_mask.size(0), diag_mask.size(1)).type(torch.ByteTensor).cuda()

        attn_scores.data.masked_fill_(attn_scores_mask, -1e10)
        attn_scores.data.masked_fill_(diag_mask, -1e10)
        attn_scores = F.softmax(attn_scores, dim=-1)

        self_attn = torch.bmm(attn_scores, key)

        ret = F.relu(self.linear_inner(torch.cat((self_attn, key, self_attn * key), dim = -1)))
        ret = ret + residual
        # query_mask_output = context_mask.unsqueeze(2).repeat(1, 1, self.args.model_dim * 2)
        query_mask_output = context_mask.unsqueeze(2).expand(context_mask.size(0), context_mask.size(1), self.args.model_dim * 2)
        ret.data.masked_fill_(query_mask_output, 0)

        return ret

class TriLinear_Attn(nn.Module):
    """
    Trilinear self attention.
    Same as https://github.com/allenai/allennlp/pull/346/files
    """
    def __init__(self, args):
        super(TriLinear_Attn, self).__init__()
        self.args = args
        self.inp_dim = self.args.model_dim * 2 + 1024 if self.args.add_elmo else self.args.model_dim * 2
        # self.LSTM = RNN_TYPES[args.rnn_type](input_size = args.model_dim * 2, hidden_size = args.model_dim, 
        #                     num_layers = 1, bidirectional = True)
        self.query_weights = nn.Parameter(torch.FloatTensor(self.inp_dim, 1))
        self.key_weights = nn.Parameter(torch.FloatTensor(self.inp_dim, 1))
        self.dot_weights = nn.Parameter(torch.FloatTensor(1, 1, self.inp_dim))
        # self.merge_linear = Linear(args.model_dim * 8, args.model_dim * 2)
        self.reset_parameters()

        self.out_LSTM = RNN_TYPES[args.rnn_type](input_size = args.model_dim * 8 + 1024 * 4 if self.args.add_elmo else self.args.model_dim * 8, hidden_size = args.model_dim, 
                            num_layers = 1, bidirectional = True)

    def reset_parameters(self):
        init.xavier_uniform(self.query_weights)
        init.xavier_uniform(self.key_weights)
        init.xavier_uniform(self.dot_weights)

    def forward(self, query_inp, query_mask, key_inp, key_mask):

        # model_output = self.LSTM(model_output)
        query_inp = F.dropout(query_inp, p = self.args.dropout, training = self.training)
        key_inp = F.dropout(key_inp, p = self.args.dropout, training = self.training)

        query = query_inp.clone()
        key = key_inp.clone()
        # value = key_inp.clone()

        batch_size, time_steps_query, inp_dim = query.size()
        _, time_steps_key, _ = key.size()

        # query (B x T x C) * (C x 1) --> (B x T x 1) 
        query_factors = torch.matmul(query.contiguous().view(batch_size*time_steps_query, inp_dim), self.query_weights)
        query_factors = query_factors.contiguous().view(batch_size, time_steps_query, 1)
        # key (B x T x C) * (C x 1) --> (B x 1 x T) 
        key_factors = torch.matmul(key.contiguous().view(batch_size*time_steps_key, inp_dim), self.key_weights)
        key_factors = key_factors.contiguous().view(batch_size, 1, time_steps_key)

        # query (B x T x C) * (1 x 1 x C) --> (B x T x C)
        query_weighted = query * self.dot_weights

        # # key (B x T x C) --> (B x C x T)
        # key_t = key.transpose(1, 2)

        # (B x Tc x C) * (B x C x Tq) --> similar to attn_scores(B x Tc x Tq)
        dot_factors = torch.matmul(query_weighted, key.transpose(1, 2))

        similarity_matrix = dot_factors + query_factors + key_factors

        # Create attn score mask
        attn_scores_mask_1 = query_mask.unsqueeze(2).expand(query_mask.size(0), query_mask.size(1), key_mask.size(1))
        attn_scores_mask_2 = key_mask.unsqueeze(1).expand(key_mask.size(0), query_mask.size(1), key_mask.size(1))
        attn_scores_mask = (attn_scores_mask_1 + attn_scores_mask_2).ge(1)

        similarity_matrix.data.masked_fill_(attn_scores_mask, -1e10)
        # attn_scores = attn_scores.data.masked_fill_(diag_mask, -1e10)
        # attn_scores = F.softmax(attn_scores, dim=-1)

        # self_attn = torch.matmul(attn_scores, value)
        c2q_attention = self.get_c2q_attention(similarity_matrix, key)
        q2c_attention = self.get_q2c_attention(similarity_matrix, query)

        result = torch.cat((query, c2q_attention, query * c2q_attention, query * q2c_attention), 2)
        # query_mask_output = query_mask.unsqueeze(2).repeat(1, 1, self.args.model_dim * 8)
        mask_dim = self.args.model_dim * 8 + 1024 * 4 if self.args.add_elmo else self.args.model_dim * 8
        query_mask_output = query_mask.unsqueeze(2).expand(query_mask.size(0), query_mask.size(1), mask_dim)
        result.data.masked_fill_(query_mask_output, 0)
        # result = F.relu(self.merge_linear(result))
        
        # replace linear&relu with LSTM
        result = self.out_LSTM(result)[0]

        query_mask_output = query_mask.unsqueeze(2).expand(query_mask.size(0), query_mask.size(1), self.args.model_dim * 2)
        result.data.masked_fill_(query_mask_output, 0)

        return result

    def get_c2q_attention(self, similarity_matrix, query_info):
    
        c2q_similarity_matrix = F.softmax(similarity_matrix, dim = -1)
        c2q_attention = torch.bmm(c2q_similarity_matrix, query_info)

        # B x Tc x C   C = model_dim * 2
        return c2q_attention 

    def get_q2c_attention(self, similarity_matrix, context_info):
        _similarity_matrix = torch.max(similarity_matrix, dim = 2)[0]
        q2c_similarity_matrix = F.softmax(_similarity_matrix, dim = 1)
        q2c_similarity_matrix = q2c_similarity_matrix.unsqueeze(1)
        
        q2c_attention = torch.bmm(q2c_similarity_matrix, context_info)
        # q2c_attention = q2c_attention.repeat(1, context_info.size(1), 1)
        q2c_attention = q2c_attention.expand(q2c_attention.size(0), context_info.size(1), q2c_attention.size(2))

        # B x Tc x C    C = model_dim * 2
        return q2c_attention 

class Self_AttentionFlow_Layer(nn.Module):
    """
    Attention class in BiDAF, according to the paper.
    """
    def __init__(self, args):
        super(Self_AttentionFlow_Layer, self).__init__()
        self.args = args
        self.linear_similarity_matrix = Linear(args.model_dim * 6, 1, bias = False)
        self.merge_linear = Linear(args.model_dim * 6, args.model_dim * 2)

    def forward(self, inp, inp_mask):

        inp = F.dropout(inp, p = self.args.dropout, training = self.training)

        context_info = inp.clone()
        query_info = inp.clone()
        context_mask = inp_mask.clone()
    #    query_mask = inp_mask.clone()
        
        similarity_matrix = self.get_similarity_matrix(context_info, context_mask)

        c2q_attention = self.get_c2q_attention(similarity_matrix, query_info)
        # q2c_attention = self.get_q2c_attention(similarity_matrix, context_info)

        # result = torch.cat((context_info, c2q_attention, context_info * c2q_attention, context_info * q2c_attention), 2)
        result = torch.cat((context_info, c2q_attention, context_info * c2q_attention), 2)

        # context_mask_output = context_mask.unsqueeze(2).repeat(1, 1, self.args.model_dim * 8)
        # context_mask_output = context_mask.unsqueeze(2).expand(context_mask.size(0), context_mask.size(1), self.args.model_dim * 6)
        # result.data.masked_fill_(context_mask_output, 0)
        result = F.relu(self.merge_linear(result))
        
        context_mask = context_mask.unsqueeze(2).expand(context_mask.size(0), context_mask.size(1), self.args.model_dim * 2)
        result.data.masked_fill_(context_mask, 0)
        
        # B x T x C    C = model_dim * 8
        return result

    def get_similarity_matrix(self, context_info, context_mask):

        tiled_context_info = context_info.unsqueeze(2).expand(context_info.size()[0],
                                                              context_info.size()[1],
                                                              context_info.size()[1],
                                                              context_info.size()[2]
                                                              )
        tiled_query_info = context_info.unsqueeze(1).expand(context_info.size()[0],
                                                          context_info.size()[1],
                                                          context_info.size()[1],
                                                          context_info.size()[2],
                                                          )
        tiled_context_mask = context_mask.unsqueeze(2).expand(context_mask.size()[0],
                                                              context_mask.size()[1],
                                                              context_mask.size()[1],
                                                              )
        tiled_query_mask = context_mask.unsqueeze(1).expand(context_mask.size()[0],
                                                          context_mask.size()[1],
                                                          context_mask.size()[1],
                                                          )
        # Get the attention mask
        attn_mask = torch.ge(tiled_context_mask + tiled_query_mask, 1)
        
        # cross_info = tiled_context_info * tiled_query_info

        concat_info = torch.cat((tiled_context_info, tiled_query_info, tiled_context_info * tiled_query_info), 3)

        # Get the high dimentional mask
        # attn_mask_concat = attn_mask.unsqueeze(3).repeat(1, 1, 1, self.args.model_dim * 6)
        # attn_mask_concat = attn_mask.unsqueeze(3).expand(attn_mask.size(0), attn_mask.size(1), attn_mask.size(2), self.args.model_dim * 6)

        # Mask the concat_info
        # concat_info.data.masked_fill_(attn_mask_concat, 0)

        similarity_matrix = self.linear_similarity_matrix(concat_info).squeeze(3)

        # Mask the final result
        similarity_matrix.data.masked_fill_(attn_mask, -1e10)
        
        # B x Tc x Tq
        return similarity_matrix

    def get_c2q_attention(self, similarity_matrix, query_info):
    
        similarity_matrix = F.softmax(similarity_matrix, dim = -1)
        c2q_attention = torch.bmm(similarity_matrix, query_info)

        # B x Tc x C   C = model_dim * 2
        return c2q_attention 

    # def get_q2c_attention(self, similarity_matrix, context_info):
    #     _similarity_matrix = torch.max(similarity_matrix, dim = 2)[0]
    #     q2c_similarity_matrix = F.softmax(_similarity_matrix, dim = 1)
    #     q2c_similarity_matrix = q2c_similarity_matrix.unsqueeze(1)
        
    #     q2c_attention = torch.bmm(q2c_similarity_matrix, context_info)
    #     # q2c_attention = q2c_attention.repeat(1, context_info.size(1), 1)
    #     q2c_attention = q2c_attention.expand(q2c_attention.size(0), context_info.size(1), q2c_attention.size(2))

    #     # B x Tc x C    C = model_dim * 2
    #     return q2c_attention 

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal(self.linear.weight)
        # g_init_fn(self.linear.weight)

    def forward(self, inputs):
        return self.linear(inputs)
