#!/usr/bin/enc python3
# Highway layer <https://arxiv.org/abs/1505.00387> that does a gate
# combination of a linear transformation and a non-layer transformation of its input.
# Wirtten by Turing Lee in ColorfulClouds .

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    """
    Highway Net : y = g * x + (1 - g) * f(A(x))
    'A' is a linear transformation
    'f' is an element-wise non-linearity 
    'g' is an element-wize gate
    """

    def __init__(self, input_dim, num_layers):
        super(Highway, self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList([nn.Linear(input_dim, input_dim * 2)
                                     for _ in range(num_layers)])
        for layer in self.layers:
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, inp):
        cur_inp = inp
        for layer in self.layers:
            linear_part = cur_inp
            cur_inp = F.dropout(cur_inp, p = 0.2, training = self.training)
            projected_inp = layer(cur_inp)
            nonlinear_part = projected_inp[:, :, (0 * self.input_dim):(1 * self.input_dim)]
            gate = projected_inp[:, :, (1 * self.input_dim):(2 * self.input_dim)]
            nonlinear_part = nn.functional.relu(nonlinear_part)
            gate = nn.functional.sigmoid(gate)
            cur_inp = gate * linear_part + (1 - gate) * nonlinear_part
        return cur_inp
    
