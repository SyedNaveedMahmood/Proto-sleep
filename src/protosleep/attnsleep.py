from __future__ import annotations

import os
import gc
import json
import math
import time
import copy
import random
import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from contextlib import nullcontext
from typing import Dict, List, Tuple, Optional, Any, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay,
)

from .config import *
from .utils import *

# =============================================================================
# 3. ATTNSLEEP -- INDEPENDENT REPRODUCTION OF THE OFFICIAL UPLOADED SOURCE
# =============================================================================
def clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class SELayer(nn.Module):
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super().__init__()
        # Equivalent to official nn.Conv1d(inplanes, planes, stride) for stride=1.
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(out + residual)
        return out


class MRCNN(nn.Module):
    def __init__(self, afr_reduced_cnn_size=30):
        super().__init__()
        drate = 0.5
        self.features1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4), nn.Dropout(drate),
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),
        )
        self.features2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50, bias=False, padding=200),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2), nn.Dropout(drate),
            nn.Conv1d(64, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.dropout = nn.Dropout(drate)
        self.inplanes = 128
        self.AFR = self._make_layer(SEBasicBlock, afr_reduced_cnn_size, 1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        x = torch.cat((x1, x2), dim=2)
        x = self.dropout(x)
        return self.AFR(x)


def scaled_dot_attention(query, key, value, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        self.__padding = (kernel_size - 1) * dilation
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                         padding=self.__padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        result = super().forward(x)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, afr_reduced_cnn_size, dropout=0.1):
        super().__init__()
        if d_model % h != 0:
            raise ValueError("d_model must be divisible by heads")
        self.d_k = d_model // h
        self.h = h
        self.convs = clones(CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size,
                                         kernel_size=7, stride=1), 3)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn = None

    def forward(self, query, key, value):
        nb = query.size(0)
        query = query.view(nb, -1, self.h, self.d_k).transpose(1, 2)
        key = self.convs[1](key).view(nb, -1, self.h, self.d_k).transpose(1, 2)
        value = self.convs[2](value).view(nb, -1, self.h, self.d_k).transpose(1, 2)
        x, self.attn = scaled_dot_attention(query, key, value, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nb, -1, self.h * self.d_k)
        return self.linear(x)


class AttnSleepLayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        # Match the official source's torch.std default semantics.
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerOutput(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = AttnSleepLayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, afr_reduced_cnn_size, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_output = clones(SublayerOutput(size, dropout), 2)
        self.size = size
        self.conv = CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size,
                                 kernel_size=7, stride=1, dilation=1)

    def forward(self, x_in):
        query = self.conv(x_in)
        x = self.sublayer_output[0](query, lambda _x: self.self_attn(query, x_in, x_in))
        return self.sublayer_output[1](x, self.feed_forward)


class TCE(nn.Module):
    def __init__(self, layer, n):
        super().__init__()
        self.layers = clones(layer, n)
        self.norm = AttnSleepLayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


def build_attnsleep_tce():
    n = 2
    d_model = 80
    d_ff = 120
    h = 5
    dropout = 0.1
    afr_size = 30
    attn = MultiHeadedAttention(h, d_model, afr_size)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    return TCE(EncoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(ff), afr_size, dropout), n)


class AttnSleepBaseline(nn.Module):
    """Model A: official-source geometry, exposed AFR for verification."""
    def __init__(self):
        super().__init__()
        self.mrcnn = MRCNN(30)
        self.tce = build_attnsleep_tce()
        self.fc = nn.Linear(80 * 30, NUM_CLASSES)

    def forward(self, x, return_features=False):
        afr = self.mrcnn(x)
        enc = self.tce(afr)
        vec = enc.contiguous().view(enc.shape[0], -1)
        logits = self.fc(vec)
        if return_features:
            return dict(logits=logits, afr=afr, encoded=enc)
        return logits


def init_attnsleep_weights(m: nn.Module):
    # Match official train_Kfold_CV.py initialization.
    if type(m) == nn.Conv1d:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.BatchNorm1d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
