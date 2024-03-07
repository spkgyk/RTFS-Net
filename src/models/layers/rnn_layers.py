import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from sru import SRU

from .attention import MultiHeadSelfAttention
from . import conv_layers, normalizations, activations


class RNNProjection(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rnn_type: str = "LSTM",
        dropout: float = 0,
        bidirectional: bool = True,
        *args,
        **kwargs,
    ):
        super(RNNProjection, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.dropout = dropout
        self.num_direction = int(bidirectional) + 1

        self.norm1 = nn.LayerNorm(self.input_size)
        self.rnn = getattr(nn, rnn_type)(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.proj = nn.Sequential(
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * self.num_direction, self.input_size),
            nn.Dropout(self.dropout),
        )
        self.norm2 = nn.LayerNorm(self.input_size)

    def forward(self, x: torch.Tensor):
        res = x
        x = x.transpose(1, 2).contiguous()

        x = self.norm1(x)
        residual = x
        self.rnn.flatten_parameters()
        x = self.rnn(x)[0]  # B, L, num_direction * H
        x = self.proj(x)
        x = self.norm2(x + residual)  # B, L, N

        x = x.transpose(1, 2).contiguous()
        x = x + res
        return x


class DualPathRNN(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        dim: int,
        kernel_size: int = 8,
        stride: int = 1,
        rnn_type: str = "LSTM",
        num_layers: int = 1,
        norm_type: str = "LayerNormalization4D",
        act_type: str = "Tanh",
        bidirectional: bool = True,
        apply_ffn: bool = False,
        *args,
        **kwargs,
    ):
        super(DualPathRNN, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.act_type = act_type
        self.bidirectional = bidirectional
        self.apply_ffn = apply_ffn

        self.num_direction = int(bidirectional) + 1
        self.unfolded_chan = self.in_chan * self.kernel_size
        self.rnn_out_chan = self.hid_chan * self.num_direction if self.rnn_type != "Attn" else self.unfolded_chan

        self.norm = normalizations.get(self.norm_type)((self.in_chan, 1) if self.norm_type == "LayerNormalization4D" else self.in_chan)
        self.unfold = nn.Unfold((self.kernel_size, 1), stride=(self.stride, 1))

        if self.rnn_type == "SRU":
            self.rnn = SRU(
                input_size=self.unfolded_chan,
                hidden_size=self.hid_chan,
                num_layers=self.num_layers,
                bidirectional=self.bidirectional,
            )
        # elif self.rnn_type == "SRUpp":
        #     self.rnn = SRUpp(
        #         input_size=self.unfolded_chan,
        #         hidden_size=self.hid_chan,
        #         proj_size=self.hid_chan,
        #         num_layers=self.num_layers,
        #         bidirectional=self.bidirectional,
        #     )
        elif self.rnn_type == "Attn":
            self.rnn = MultiHeadSelfAttention(self.unfolded_chan, 8, batch_first=False)
        else:
            self.rnn = getattr(nn, self.rnn_type)(
                input_size=self.unfolded_chan,
                hidden_size=self.hid_chan,
                num_layers=self.num_layers,
                bidirectional=self.bidirectional,
            )

        self.ffn = (
            conv_layers.FeedForwardNetwork(self.unfolded_chan, self.unfolded_chan * 2, self.kernel_size, dropout=0.1)
            if self.apply_ffn
            else nn.Identity()
        )
        self.linear = nn.ConvTranspose1d(self.rnn_out_chan, self.in_chan, self.kernel_size, stride=self.stride)
        # self.linear = nn.Sequential(
        #     nn.ConvTranspose1d(self.rnn_out_chan, self.rnn_out_chan, self.kernel_size, stride=self.stride, groups=self.rnn_out_chan),
        #     activations.get(self.act_type)(),
        #     nn.Conv1d(self.rnn_out_chan, self.in_chan, 1),
        # )

    def forward(self, x: torch.Tensor):
        if self.dim == 4:
            x = x.transpose(-2, -1).contiguous()

        B, C, old_T, old_F = x.shape
        new_T = math.ceil((old_T - self.kernel_size) / self.stride) * self.stride + self.kernel_size
        new_F = math.ceil((old_F - self.kernel_size) / self.stride) * self.stride + self.kernel_size
        x = F.pad(x, (0, new_F - old_F, 0, new_T - old_T))

        residual = x
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous().view(B * new_F, C, new_T, 1)
        x = self.unfold(x)  # B * new_F, C * kernel_size, unfolded_T
        x = x.permute(2, 0, 1)  # unfolded_T, B * new_F, C * kernel_size
        x = self.rnn(x)[0] if self.rnn_type != "Attn" else self.rnn(x)
        x = x.permute(1, 2, 0)
        x = self.ffn(x)
        x = self.linear(x)
        x = x.view([B, new_F, C, new_T])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x + residual
        x = x[..., :old_T, :old_F]

        if self.dim == 4:
            x = x.transpose(-2, -1).contiguous()

        return x


class ConvLSTMCell(nn.Module):
    def __init__(self, in_chan: int, hid_chan: int, kernel_size: int = 1, num_directions: int = 1, *args, **kwargs):
        super(ConvLSTMCell, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.num_directions = num_directions

        self.linear_ih = nn.Sequential(
            conv_layers.ConvActNorm(self.in_chan, self.in_chan, self.kernel_size, groups=self.in_chan),
            conv_layers.ConvActNorm(self.in_chan, 4 * self.hid_chan, 1),
        )
        self.linear_hh = conv_layers.ConvActNorm(self.hid_chan, 4 * self.hid_chan, 1)

        if self.num_directions > 1:
            self.linear_ih_b = nn.Sequential(
                conv_layers.ConvActNorm(self.in_chan, self.in_chan, self.kernel_size, groups=self.in_chan),
                conv_layers.ConvActNorm(self.in_chan, 4 * self.hid_chan, 1),
            )
            self.linear_hh_b = conv_layers.ConvActNorm(self.hid_chan, 4 * self.hid_chan, 1)

    def forward(self, input: torch.Tensor, hidden_t: torch.Tensor, cell_t: torch.Tensor):
        # x has size: (B, C, L)
        batch_size = input.shape[0]

        if self.num_directions > 1:
            input_f, input_b = input.chunk(2, 1)
            hidden_t_f, hidden_t_b = hidden_t.chunk(2, 1)
            gates_f = self.linear_ih(input_f) + self.linear_hh(hidden_t_f)[:batch_size]
            gates_b = self.linear_ih_b(input_b) + self.linear_hh_b(hidden_t_b)[:batch_size]
            gates = torch.cat((gates_f, gates_b), dim=1)
        else:
            gates = self.linear_ih(input) + self.linear_hh(hidden_t)[:batch_size]

        i_t, f_t, g_t, o_t = gates.chunk(4, 1)

        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        g_t = torch.tanh(g_t)
        o_t = torch.sigmoid(o_t)

        c_next = (f_t * cell_t[:batch_size]) + (i_t * g_t)
        h_next = o_t * torch.tanh(c_next)

        return h_next, c_next


class BiLSTM2D(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        dim: int = 3,
        kernel_size: int = 5,
        window: int = 8,
        stride: int = 1,
        act_type: str = "PReLU",
        norm_type: str = "gLN",
        bidirectional: bool = True,
        *args,
        **kwargs,
    ):
        super(BiLSTM2D, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.dim = dim
        self.kernel_size = kernel_size
        self.window = window
        self.stride = stride
        self.act_type = act_type
        self.norm_type = norm_type
        self.bidirectional = bidirectional

        self.num_dir = int(bidirectional) + 1

        self.norm = normalizations.get(self.norm_type)(self.in_chan)
        self.lstm_cell = ConvLSTMCell(self.in_chan * self.window, self.hid_chan, self.kernel_size, self.num_dir)
        ks = (self.window, 1)
        s = (self.stride, 1)
        self.unfold = nn.Unfold(kernel_size=ks, stride=s)
        self.projection = nn.Sequential(
            nn.ConvTranspose2d(
                self.hid_chan * self.num_dir,
                self.hid_chan * self.num_dir,
                ks,
                groups=self.hid_chan * self.num_dir,
                stride=s,
            ),
            activations.get(self.act_type)(),
            normalizations.get(self.norm_type)(self.hid_chan * self.num_dir),
            conv_layers.ConvActNorm(self.hid_chan * self.num_dir, self.in_chan, 1, is2d=True),
        )
        # make this depth wise seperable

    def pad(self, x: torch.Tensor):
        old_w, old_h = x.shape[-2:]
        new_w = math.ceil((old_w - self.window) / self.stride) * self.stride + self.window
        new_h = math.ceil((old_h - self.window) / self.stride) * self.stride + self.window
        x = F.pad(x, (0, new_h - old_h, 0, new_w - old_w))

        iterations = math.ceil(new_h / self.window)

        return x, old_w, old_h, iterations

    def init_states(self, x: torch.Tensor):
        hidden_t = torch.zeros((1, self.hid_chan * self.num_dir, 1), device=x.device)
        cell_t = torch.zeros((1, self.hid_chan * self.num_dir, 1), device=x.device)

        return hidden_t, cell_t

    def forward(self, x: torch.Tensor):
        # x has size: (B, C, T, F)
        bs = x.shape[0]

        residual = x
        x = self.norm(x)

        x = torch.cat((x, x.flip(self.dim - 1)), dim=1) if self.bidirectional else x
        x = x.transpose(-1, -2).contiguous() if self.dim == 3 else x

        x, old_w, old_h, iterations = self.pad(x)
        hidden_t, cell_t = self.init_states(x)

        outputs = [None] * iterations
        for i in range(iterations):
            x_slice = x[..., i * self.window : (i + 1) * self.window]
            w, h = x_slice.shape[-2:]
            x_slice = x_slice.permute(0, 3, 1, 2).contiguous().view(bs * h, self.in_chan * self.num_dir, w, 1)
            x_slice = self.unfold(x_slice)
            hidden_t, cell_t = self.lstm_cell(x_slice, hidden_t, cell_t)
            outputs[i] = hidden_t.view(bs, h, self.hid_chan * self.num_dir, -1).permute(0, 2, 3, 1).contiguous()

        x = self.projection(torch.cat(outputs, dim=-1))[..., :old_w, :old_h]
        x = x.transpose(-1, -2).contiguous() if self.dim == 3 else x
        x = x + residual

        return x


class GlobalAttentionRNN(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int = None,
        dropout: float = 0.1,
        rnn_type: str = "LSTM",
        bidirectional: bool = True,
        *args,
        **kwargs,
    ):
        super(GlobalAttentionRNN, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan if hid_chan is not None else self.in_chan
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

        self.RNN = RNNProjection(self.in_chan, self.hid_chan, self.rnn_type, self.dropout, self.bidirectional)

    def forward(self, x: torch.Tensor):
        x = self.RNN(x)
        return x


class GlobalGALR(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int = None,
        ffn_name: str = "FeedForwardNetwork",
        kernel_size: int = 5,
        n_head: int = 8,
        dropout: float = 0.1,
        group_ffn: bool = False,
        pos_enc: bool = True,
        rnn_type: str = "LSTM",
        bidirectional: bool = True,
        *args,
        **kwargs,
    ):
        super(GlobalGALR, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan if hid_chan is not None else 2 * self.in_chan
        self.ffn_name = ffn_name
        self.kernel_size = kernel_size
        self.n_head = n_head
        self.dropout = dropout
        self.group_ffn = group_ffn
        self.pos_enc = pos_enc
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

        self.time_RNN = RNNProjection(self.in_chan, self.in_chan, self.rnn_type, self.dropout, self.bidirectional)
        self.freq_MHSA = MultiHeadSelfAttention(self.in_chan, self.n_head, self.dropout, self.pos_enc)
        self.freq_FFN = conv_layers.get(ffn_name)(self.in_chan, self.hid_chan, self.kernel_size, dropout=dropout)

        self.group_FFN = nn.Identity()
        if self.group_ffn:
            self.group_FFN = conv_layers.FeedForwardNetwork(self.in_chan, self.hid_chan, self.kernel_size, dropout=dropout, is2d=True)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.size()

        x = x.permute(0, 3, 1, 2).contiguous().view(B * W, C, H)
        x = self.time_RNN.forward(x)
        x = x.view(B, W, C, H).permute(0, 2, 3, 1).contiguous()

        x = x.permute(0, 2, 1, 3).contiguous().view(B * H, C, W)
        x = self.freq_MHSA.forward(x)
        x = self.freq_FFN.forward(x)
        x = x.view(B, H, C, W).permute(0, 2, 1, 3).contiguous()

        x = self.group_FFN(x)

        return x
