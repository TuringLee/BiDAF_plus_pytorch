#!/usr/bin/enc python3
# BiDAF model of Jarvis for SQuAD .
# Wirtten by Turing Lee in ColorfulClouds .

import numpy as np
import torch
from torch import nn
from torch.nn import init

class PosEncoding(nn.Module):
    def __init__(self, max_seq_len, d_word_vec):
        super(PosEncoding, self).__init__()
        pos_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / d_word_vec) for j in range(d_word_vec)]
            if pos != 0 else np.zeros(d_word_vec) for pos in range(max_seq_len)])
        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])
        # pos_enc = np.concatenate([np.zeros([1, d_word_vec]).astype(np.float32), pos_enc])

        # additional one row for PAD idx
        self.pos_enc = nn.Embedding(max_seq_len, d_word_vec, padding_idx=1)
        # fixed positional encoding
        self.pos_enc.weight = nn.Parameter(torch.from_numpy(pos_enc).float(), False)

    def forward(self, positions):

        return self.pos_enc(positions)

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal(self.linear.weight)
        # g_init_fn(self.linear.weight)

    def forward(self, inputs):
        return self.linear(inputs)

def make_positions(batch_size, seq_len):
    positions = torch.arange(1, seq_len+1)
    positions = positions.unsqueeze(0).expand(batch_size, seq_len)
    return positions