#!/usr/bin/enc python3
# Character embedding's dimension is B x Tc x word_size x embedding size
# So Conv2d is used for further process the character embedding.
# Wirtten by Turing Lee in ColorfulClouds .

import torch.nn as nn
import torch.nn.functional as F

def multi_conv1d(inp, out_channel, kernel_size):
    outs = []
    inp = inp.permute(0, 3, 1, 2)
    
