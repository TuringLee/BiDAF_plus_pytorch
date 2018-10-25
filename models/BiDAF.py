#!/usr/bin/enc python3
# BiDAF model of Jarvis for SQuAD .
# Wirtten by Turing Lee in ColorfulClouds .

import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from allennlp.modules.elmo import Elmo, batch_to_ids
from .Highway import Highway
from .Self_Attn_Layers import Multi_Head_Self_Attn, TriLinear_Self_Attn, TriLinear_Attn, Self_AttentionFlow_Layer, MultiHeadedAttention
from .Modules import Linear

logger = logging.getLogger(__name__)

RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

class BiDAF_model(nn.Module):
    # nerual network of BiDAF
    def __init__(self, args, encoding_layer, attentionflow_layer, model_layer, output_layer):
        super(BiDAF_model, self).__init__()
        self.args = args
        self.encoding_layer = encoding_layer
        self.attentionflow_layer = attentionflow_layer
        self.model_layer = model_layer                  # if use self_attn, model_layer == self_attn_layer else 2 layers bi-LSTM
        self.output_layer = output_layer

    def forward(self, context_info, context_feature, context_mask, query_info, query_feature, query_mask, context_char, query_char, context_words, question_words):
        # pytorch 0.3.1 mask must be ByteTensor not Variable
        context_mask = context_mask.data
        query_mask = query_mask.data

        context_encoding, query_encoding = self.encoding_layer(context_info, context_feature, context_mask, query_info, query_feature, query_mask, context_char, query_char, context_words, question_words)
        attentionflow = self.attentionflow_layer(context_encoding, context_mask, query_encoding, query_mask)
        
        # XD multi_head attention
        self_attn_mask = self.get_self_attn_mask(context_mask)
        model_info = self.model_layer(attentionflow, self_attn_mask)
 
        start_prob, end_prob = self.output_layer(model_info, context_mask)
 
        return start_prob, end_prob

    def get_self_attn_mask(self, context_mask):
        batch_size, time_steps = context_mask.size()
        context_mask_similarities_1 = context_mask.unsqueeze(2).expand(batch_size, time_steps, time_steps)
        context_mask_similarities_2 = context_mask.unsqueeze(1).expand(batch_size, time_steps, time_steps)

        diag_mask = torch.eye(time_steps)
        diag_mask = diag_mask.unsqueeze(0).expand(batch_size, time_steps, time_steps).type(torch.ByteTensor).cuda()
        
        context_mask_similarities = torch.ge(context_mask_similarities_1 + context_mask_similarities_2 + diag_mask, 1)
        # context_mask_similarities = context_mask_similarities.repeat(num_heads, 1, 1)
        # context_mask_similarities = context_mask_similarities.unsqueeze(1).expand(batch_size, num_heads, time_steps, time_steps)

        return context_mask_similarities


class Encoding_Layer(nn.Module):
    """
    Encoding layer of BiDAF
    """
    def __init__(self, args, word_dict, char_dict):
        super(Encoding_Layer, self).__init__()
        self.args = args
        self.word_dict = word_dict
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim, padding_idx = 0)
        if args.use_char_emb:
            self.embedding_char = nn.Embedding(len(char_dict), args.char_emb_dim , padding_idx = 0)
            self.conv = nn.Conv2d(args.char_emb_dim, args.out_char_dim, kernel_size = (1, args.cnn_kernel_size))
            self.conv.weight.data.uniform_(-0.05, 0.05)
            highway_in_dim = args.embedding_dim + args.out_char_dim
        else:
            highway_in_dim = args.embedding_dim

        if args.add_elmo:
            highway_in_dim += 1024
       
        # self.highway = Highway(highway_in_dim, args.num_highways)
        # self.highway_query = Highway(highway_in_dim, args.num_highways)
        if args.add_features:
            self.RNN_context = RNN_TYPES[args.rnn_type](input_size = highway_in_dim + args.num_features, hidden_size = args.model_dim,
                                       num_layers = 1, bidirectional = True)
        else:
            self.RNN_context = RNN_TYPES[args.rnn_type](input_size = highway_in_dim, hidden_size = args.model_dim,
                                      num_layers = 1, bidirectional = True)
            for p in self.RNN_context.parameters():
                p.data.normal_(0, 0.05)
        if not args.share_lstm_weight:
            self.RNN_query = RNN_TYPES[args.rnn_type](input_size = highway_in_dim, hidden_size = args.model_dim,
                                      num_layers = 1, bidirectional = True)

        if args.add_elmo:
            self.elmo = Elmo(args.elmo_options_file, args.elmo_weight_file, 1, dropout = 0)
        # if args.embedding_file:
        #     self.load_embedding(word_dict, args.embedding_file)

        # if self.args.residual:
        #     self.linear_res = Linear(highway_in_dim, args.model_dim * 2, bias = False)


    def forward(self, context_info, context_feature, context_mask, query_info, query_feature, query_mask, context_char, query_char, context_words, query_words):
        """
        Inputs:
            context: document word indices [batch * len_d]
            context_feature: document word feature indices [batch * len_d * nfeat]
            context_mask: document padding mask
            query: question word indices
            query_mask: question padding mask
        """

        # embedding both context and query
        context_info = self.embedding(context_info)
        query_info = self.embedding(query_info)
        
        if self.args.use_char_emb:
            context_char_emb = self.charater_embedding(context_char)
            query_char_emb = self.charater_embedding(query_char)

            context_info = torch.cat((context_info, context_char_emb), dim = 2)
            query_info = torch.cat((query_info, query_char_emb), dim = 2)

        # context_info = self.highway(context_info)
        # query_info = self.highway(query_info)
        
        # Add manual features
        if self.args.add_features:
            context_info = torch.cat((context_info, context_feature), 2)
            query_info = torch.cat((query_info, query_feature), 2)

        context_info = F.dropout(context_info, p = self.args.dropout, training = self.training)
        query_info = F.dropout(query_info, p = self.args.dropout, training = self.training)

        if self.args.add_elmo:
            context_elmo_ids = batch_to_ids(context_words).cuda()
            context_elmo = self.elmo(context_elmo_ids)['elmo_representations'][-1]
            query_elmo_ids = batch_to_ids(query_words).cuda()
            query_elmo = self.elmo(query_elmo_ids)['elmo_representations'][-1]

            context_elmo = F.dropout(context_elmo, p = self.args.elmo_dropout, training = self.training)
            query_elmo = F.dropout(query_elmo, p = self.args.elmo_dropout, training = self.training)

            context_info = torch.cat((context_info, context_elmo), dim = 2)
            query_info = torch.cat((query_info, query_elmo), dim = 2)

        # B x T x C -> T x B x C
        context_info = context_info.transpose(0, 1)
        query_info = query_info.transpose(0, 1) 
        
        # Encoding the information
        # if self.args.residual:
        #     context_res = self.linear_res(context_info)
        #     query_res = self.linear_res(query_info)

        context_info, _ = self.RNN_context(context_info)
        if self.args.share_lstm_weight:
            query_info, _ = self.RNN_context(query_info)
        else:
            query_info, _ = self.RNN_query(query_info)
        # T x B x C -> B x T x C
        # if self.args.residual:
        #     context_info = (context_res + context_info) * math.sqrt(0.5)
        #     query_info = (query_res + query_info) * math.sqrt(0.5)

        context_info = context_info.transpose(0, 1)
        query_info = query_info.transpose(0, 1)

        if self.args.add_elmo:
            context_info = torch.cat((context_info, context_elmo), dim = 2)
            query_info = torch.cat((query_info, query_elmo), dim = 2)        

        # Masking the output
        # context_mask = context_mask.unsqueeze(2).repeat(1, 1, self.args.model_dim * 2)
        # query_mask = query_mask.unsqueeze(2).repeat(1, 1, self.args.model_dim * 2)
        mask_dim = self.args.model_dim * 2 + 1024 if self.args.add_elmo else self.args.model_dim * 2

        context_mask = context_mask.unsqueeze(2).expand(context_mask.size(0), context_mask.size(1), mask_dim)
        query_mask = query_mask.unsqueeze(2).expand(query_mask.size(0), query_mask.size(1), mask_dim)
        
        context_info.data.masked_fill_(context_mask, 0)
        query_info.data.masked_fill_(query_mask, 0)

        return context_info, query_info

    def charater_embedding(self, chars):
        batch_size, time_step, word_th = chars.size()
        chars = chars.view(batch_size * time_step, -1)
        char_emb = self.embedding_char(chars)
        char_emb = char_emb.view(batch_size, time_step, word_th, self.args.char_emb_dim)
        char_emb = F.dropout(char_emb, p = self.args.dropout, training = self.training)
        char_emb = char_emb.permute(0, 3, 1, 2)
        char_emb = self.conv(char_emb)
        char_emb = F.relu(char_emb.permute(0, 2, 3, 1))
        char_emb = torch.max(char_emb, 2)[0].squeeze(2)
        return char_emb


class AttentionFlow_Layer(nn.Module):
    """
    Attention class in BiDAF, according to the paper.
    """
    def __init__(self, args):
        super(AttentionFlow_Layer, self).__init__()
        self.args = args
        self.linear_similarity_matrix = Linear(args.model_dim * 6, 1, bias = False)
        self.merge_linear = Linear(args.model_dim * 8, args.model_dim * 2)

    def forward(self, context_info, context_mask, query_info, query_mask):
        
        context_info = F.dropout(context_info, p = self.args.dropout, training = self.training)
        query_info = F.dropout(query_info, p = self.args.dropout, training = self.training)

        similarity_matrix = self.get_similarity_matrix(context_info, context_mask, query_info, query_mask)

        c2q_attention = self.get_c2q_attention(similarity_matrix, query_info)
        q2c_attention = self.get_q2c_attention(similarity_matrix, context_info)

        result = torch.cat((context_info, c2q_attention, context_info * c2q_attention, context_info * q2c_attention), 2)
        # context_mask_output = context_mask.unsqueeze(2).repeat(1, 1, self.args.model_dim * 8)
        # context_mask_output = context_mask.unsqueeze(2).expand(context_mask.size(0), context_mask.size(1), self.args.model_dim * 8)
        # result.data.masked_fill_(context_mask_output, 0)
        context_mask = context_mask.unsqueeze(2).expand(context_mask.size(0), context_mask.size(1), self.args.model_dim * 8)
        result.data.masked_fill_(context_mask, 0)
        result = F.relu(self.merge_linear(result))
        # B x T x C    C = model_dim * 8
        return result

    def get_similarity_matrix(self, context_info, context_mask, query_info, query_mask):

        tiled_context_info = context_info.unsqueeze(2).expand(context_info.size()[0],
                                                              context_info.size()[1],
                                                              query_info.size()[1],
                                                              context_info.size()[2]
                                                              )
        tiled_query_info = query_info.unsqueeze(1).expand(query_info.size()[0],
                                                          context_info.size()[1],
                                                          query_info.size()[1],
                                                          query_info.size()[2],
                                                          )
        tiled_context_mask = context_mask.unsqueeze(2).expand(context_mask.size()[0],
                                                              context_mask.size()[1],
                                                              query_info.size()[1],
                                                              )
        tiled_query_mask = query_mask.unsqueeze(1).expand(query_mask.size()[0],
                                                          context_mask.size()[1],
                                                          query_mask.size()[1],
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
    
        c2q_similarity_matrix = F.softmax(similarity_matrix, dim = -1)
        c2q_attention = torch.bmm(c2q_similarity_matrix, query_info)

        # B x Tc x C   C = model_dim * 2
        return c2q_attention 

    def get_q2c_attention(self, similarity_matrix, context_info):
        similarity_matrix = torch.max(similarity_matrix, dim = 2)[0]
        q2c_similarity_matrix = F.softmax(similarity_matrix, dim = 1)
        q2c_similarity_matrix = q2c_similarity_matrix.unsqueeze(1)
        
        q2c_attention = torch.bmm(q2c_similarity_matrix, context_info)
        # q2c_attention = q2c_attention.repeat(1, context_info.size(1), 1)
        q2c_attention = q2c_attention.expand(q2c_attention.size(0), context_info.size(1), q2c_attention.size(2))

        # B x Tc x C    C = model_dim * 2
        return q2c_attention 

class Model_layer(nn.Module):
    """
    Modeling Layer, encodes the query-aware representations of context words.
    """
    def __init__(self, args):
        super(Model_layer, self).__init__()
        self.args = args        
        # self.linear = Linear()
        self.LSTMs = nn.ModuleList()
        self.LSTMs.append(RNN_TYPES[args.rnn_type](input_size = args.model_dim * 2, hidden_size = args.model_dim, 
                           num_layers = 1, bidirectional = True))
        for i in range(args.model_lstm_layers-1):
            self.LSTMs.append(RNN_TYPES[args.rnn_type](input_size = args.model_dim * 2, hidden_size = args.model_dim, 
                           num_layers = 1, bidirectional = True))


    def forward(self, query_aware_representation, context_mask):
        
        inp = query_aware_representation.transpose(0, 1)
        context_mask_model = context_mask.unsqueeze(2).expand(context_mask.size(0), context_mask.size(1), self.args.model_dim * 2).transpose(0, 1)

        for i in range(self.args.model_lstm_layers):
            # residual = inp
            inp = F.dropout(inp, p = self.args.dropout, training = self.training)
            inp, _ = self.LSTMs[i](inp)
            # inp = inp + residual
            inp.data.masked_fill_(context_mask_model, 0)
        
        output = inp.transpose(0, 1)
        return output

class Output_layer(nn.Module):
    """
    Output Layer to predict the span.
    """
    def __init__(self, args):
        super(Output_layer, self).__init__()
        self.args = args
        self.RNN_start = RNN_TYPES[args.rnn_type](input_size = args.model_dim * 2, hidden_size = args.model_dim, 
                                 num_layers = 1, bidirectional = True)
        
        for p in self.RNN_start.parameters():
            p.data.normal_(0, 0.05)
        self.RNN_end = RNN_TYPES[args.rnn_type](input_size = args.model_dim * 4, hidden_size = args.model_dim, 
                                 num_layers = 1, bidirectional = True)
        for p in self.RNN_end.parameters():
            p.data.normal_(0, 0.05)
        self.linear_start = Linear(args.model_dim * 2, 1, bias = False)
        self.linear_end = Linear(args.model_dim * 2, 1, bias = False)
        # self.LSTMs = nn.ModuleList()
        # for i in range(args.output_lstm_layers):
        #     self.LSTMs.append(RNN_TYPES[args.rnn_type](input_size = args.model_dim * 2, hidden_size = args.model_dim, 
        #                       num_layers = 1, bidirectional = True))

    # def forward(self, Attention_output, Self_attn, context_mask):
    def forward(self, Self_attn, context_mask):
        
        # context_mask_model = context_mask.unsqueeze(2).expand(context_mask.size(0), context_mask.size(1), self.args.model_dim * 2)
        start_inp = F.dropout(Self_attn, p = self.args.dropout, training = self.training)
        # start_inp = torch.cat((Attention_output, Model_output), 2)
        # start_inp = Attention_output + Self_attn
        # start_inp = Self_attn
        # start_inp = F.dropout(start_inp, p = self.args.dropout, training = self.training)
        #start_inp_ = start_inp.transpose(0, 1)
        start_rep = self.RNN_start(start_inp.transpose(0, 1))[0].transpose(0, 1)
        #start_rep = start_rep.transpose(0, 1)
        start_prob = self.linear_start(start_rep).squeeze(2)
        # start_prob = F.dropout(start_prob, p = self.args.dropout, training = self.training)
        start_prob.data.masked_fill_(context_mask, -1e08)
        
        # Model_output = Model_output.transpose(0, 1)
        
        # for i in range(self.args.output_lstm_layers):
        #     residual = Model_output
        #     Model_output = F.dropout(Model_output, p = self.args.dropout, training = self.training)
        #     Model_output, _ = self.LSTMs[i](Model_output)
        #     if self.args.residual:
        #         Model_output = (residual + Model_output) * math.sqrt(0.5)
        #     Model_output.data.masked_fill_(context_mask_model, 0)
        # Model_output = Model_output.transpose(0, 1)

        end_inp = torch.cat((start_inp, start_rep), 2)
        # end_inp = F.dropout(end_inp, p = self.args.dropout, training = self.training)
        #end_inp_ = end_inp.transpose(0, 1)
        end_rep = self.RNN_end(end_inp.transpose(0, 1))[0].transpose(0, 1)
        #end_rep = end_rep.transpose(0, 1)
        end_prob = self.linear_end(end_rep).squeeze(2)
        # end_prob = F.dropout(end_prob, p = self.args.dropout, training = self.training)
        end_prob.data.masked_fill_(context_mask, -1e08)
        
        if self.training:
            start_prob = F.log_softmax(start_prob, dim = -1)
            end_prob = F.log_softmax(end_prob, dim = -1)
        else:
            start_prob = F.softmax(start_prob, dim = -1)
            end_prob = F.softmax(end_prob, dim = -1)
        
        return start_prob, end_prob

def build_model(args, word_dict, char_dict, normalize):
    encoding_layer = Encoding_Layer(args, word_dict, char_dict)
    # attentionflow_layer = AttentionFlow_Layer(args)
    attentionflow_layer = TriLinear_Attn(args)
    if not args.self_attn:
        model_layer = Model_layer(args)
    output_layer = Output_layer(args)
    if args.self_attn:
        # self_attn_layer = Self_AttentionFlow_Layer(args)
        # model_layer = TriLinear_Self_Attn(args)
        model_layer = MultiHeadedAttention(args)
        return BiDAF_model(args, encoding_layer, attentionflow_layer, model_layer, output_layer)
    
    return BiDAF_model(args, encoding_layer, attentionflow_layer, model_layer, output_layer)



