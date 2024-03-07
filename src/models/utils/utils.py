import torch
import numpy as np
import torch.nn as nn

from thop import profile
from torch.autograd import Variable


def pad_segment(input, block_size):
    # input is the features: (B, N, T)
    batch_size, dim, seq_len = input.shape
    block_stride = block_size // 2

    rest = block_size - (block_stride + seq_len % block_size) % block_size
    if rest > 0:
        pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type()).to(input.device)
        input = torch.cat([input, pad], 2)

    pad_aux = Variable(torch.zeros(batch_size, dim, block_stride)).type(input.type()).to(input.device)
    input = torch.cat([pad_aux, input, pad_aux], 2)

    return input, rest


def split_feature(input, block_size):
    # split the feature into chunks of segment size
    # input is the features: (B, N, T)

    input, rest = pad_segment(input, block_size)
    batch_size, dim, seq_len = input.shape
    block_stride = block_size // 2

    block1 = input[:, :, :-block_stride].contiguous().view(batch_size, dim, -1, block_size)
    block2 = input[:, :, block_stride:].contiguous().view(batch_size, dim, -1, block_size)
    block = torch.cat([block1, block2], 3).view(batch_size, dim, -1, block_size).transpose(2, 3)

    return block.contiguous(), rest


def merge_feature(input, rest):
    # merge the splitted features into full utterance
    # input is the features: (B, N, L, K)

    batch_size, dim, block_size, _ = input.shape
    block_stride = block_size // 2
    input = input.transpose(2, 3).contiguous().view(batch_size, dim, -1, block_size * 2)  # B, N, K, L

    input1 = input[:, :, :, :block_size].contiguous().view(batch_size, dim, -1)[:, :, block_stride:]
    input2 = input[:, :, :, block_size:].contiguous().view(batch_size, dim, -1)[:, :, :-block_stride]

    output = input1 + input2
    if rest > 0:
        output = output[:, :, :-rest]

    return output.contiguous()  # B, N, T


def get_bandwidths(win: int, sr: int = 16000):
    enc_dim = win // 2 + 1

    bandwidth_100 = int(np.floor(100 / (sr / 2.0) * enc_dim))
    bandwidth_250 = int(np.floor(250 / (sr / 2.0) * enc_dim))
    bandwidth_500 = int(np.floor(500 / (sr / 2.0) * enc_dim))
    bandwidth_1k = int(np.floor(1000 / (sr / 2.0) * enc_dim))
    band_width = [bandwidth_100] * 5
    band_width += [bandwidth_250] * 6
    band_width += [bandwidth_500] * 4
    band_width += [bandwidth_1k] * 4
    if sr > 160000:
        bandwidth_2k = int(np.floor(2000 / (sr / 2.0) * enc_dim))
        band_width += [bandwidth_2k] * 1

    assert enc_dim > np.sum(band_width), f"{(enc_dim)}, {np.sum(band_width)} = sum({band_width})"

    band_width.append(enc_dim - np.sum(band_width))

    return band_width


def get_MACS_params(layer: nn.Module, inputs: tuple = None):
    macs = None
    if inputs is not None:
        macs = int(profile(layer, inputs=inputs, verbose=False)[0] / 1000000)
    params = int(sum(p.numel() for p in layer.parameters() if p.requires_grad) / 1000)

    return [macs, params]
