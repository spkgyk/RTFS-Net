import torch
import torch.nn as nn
import torch.nn.functional as functional

from . import conv_layers
from timm.models.layers import DropPath


class PositionalEncoding(nn.Module):
    def __init__(self, channels: int, max_len: int = 10000, *args, **kwargs):
        super(PositionalEncoding, self).__init__()
        self.channels = channels
        self.max_len = max_len

        pe = torch.zeros(self.max_len, self.channels, requires_grad=False)
        position = torch.arange(0, self.max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.channels, 2).float() * -(torch.log(torch.tensor(self.max_len).float()) / self.channels))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        in_chan: int,
        n_head: int = 8,
        dropout: int = 0.1,
        positional_encoding: bool = True,
        batch_first=True,
        *args,
        **kwargs,
    ):
        super(MultiHeadSelfAttention, self).__init__()
        self.in_chan = in_chan
        self.n_head = n_head
        self.dropout = dropout
        self.positional_encoding = positional_encoding
        self.batch_first = batch_first

        assert self.in_chan % self.n_head == 0, "In channels: {} must be divisible by the number of heads: {}".format(
            self.in_chan, self.n_head
        )

        self.norm1 = nn.LayerNorm(self.in_chan)
        self.pos_enc = PositionalEncoding(self.in_chan) if self.positional_encoding else nn.Identity()
        self.attention = nn.MultiheadAttention(self.in_chan, self.n_head, self.dropout, batch_first=self.batch_first)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.norm2 = nn.LayerNorm(self.in_chan)
        self.drop_path_layer = DropPath(self.dropout)

    def forward(self, x: torch.Tensor):
        res = x
        if self.batch_first:
            x = x.transpose(1, 2)  # B, C, T -> B, T, C

        x = self.norm1(x)
        x = self.pos_enc(x)
        residual = x
        x = self.attention(x, x, x)[0]
        x = self.dropout_layer(x) + residual
        x = self.norm2(x)

        if self.batch_first:
            x = x.transpose(2, 1)  # B, T, C -> B, C, T

        x = self.drop_path_layer(x) + res
        return x


class MultiHeadSelfAttention2D(nn.Module):
    def __init__(
        self,
        in_chan: int,
        n_freqs: int,
        n_head: int = 4,
        hid_chan: int = 4,
        act_type: str = "PReLU",
        norm_type: str = "LayerNormalization4D",
        dim: int = 3,
        *args,
        **kwargs,
    ):
        super(MultiHeadSelfAttention2D, self).__init__()
        self.in_chan = in_chan
        self.n_freqs = n_freqs
        self.n_head = n_head
        self.hid_chan = hid_chan
        self.act_type = act_type
        self.norm_type = norm_type
        self.dim = dim

        assert self.in_chan % self.n_head == 0

        self.Queries = nn.ModuleList()
        self.Keys = nn.ModuleList()
        self.Values = nn.ModuleList()

        for _ in range(self.n_head):
            self.Queries.append(
                conv_layers.ConvActNorm(
                    in_chan=self.in_chan,
                    out_chan=self.hid_chan,
                    kernel_size=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    n_freqs=self.n_freqs,
                    is2d=True,
                )
            )
            self.Keys.append(
                conv_layers.ConvActNorm(
                    in_chan=self.in_chan,
                    out_chan=self.hid_chan,
                    kernel_size=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    n_freqs=self.n_freqs,
                    is2d=True,
                )
            )
            self.Values.append(
                conv_layers.ConvActNorm(
                    in_chan=self.in_chan,
                    out_chan=self.in_chan // self.n_head,
                    kernel_size=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    n_freqs=self.n_freqs,
                    is2d=True,
                )
            )

        self.attn_concat_proj = conv_layers.ConvActNorm(
            in_chan=self.in_chan,
            out_chan=self.in_chan,
            kernel_size=1,
            act_type=self.act_type,
            norm_type=self.norm_type,
            n_freqs=self.n_freqs,
            is2d=True,
        )

    def forward(self, x: torch.Tensor):
        if self.dim == 4:
            x = x.transpose(-2, -1).contiguous()

        batch_size, _, time, freq = x.size()
        residual = x

        all_Q = [q(x) for q in self.Queries]  # [B, E, T, F]
        all_K = [k(x) for k in self.Keys]  # [B, E, T, F]
        all_V = [v(x) for v in self.Values]  # [B, C/n_head, T, F]

        Q = torch.cat(all_Q, dim=0)  # [B', E, T, F]    B' = B*n_head
        K = torch.cat(all_K, dim=0)  # [B', E, T, F]
        V = torch.cat(all_V, dim=0)  # [B', C/n_head, T, F]

        Q = Q.transpose(1, 2).flatten(start_dim=2)  # [B', T, E*F]
        K = K.transpose(1, 2).flatten(start_dim=2)  # [B', T, E*F]
        V = V.transpose(1, 2)  # [B', T, C/n_head, F]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*F/n_head]
        emb_dim = Q.shape[-1]  # C*F/n_head

        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)  # [B', T, T]
        attn_mat = functional.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*F/n_head]
        V = V.reshape(old_shape)  # [B', T, C/n_head, F]
        V = V.transpose(1, 2)  # [B', C/n_head, T, F]
        emb_dim = V.shape[1]  # C/n_head

        x = V.view([self.n_head, batch_size, emb_dim, time, freq])  # [n_head, B, C/n_head, T, F]
        x = x.transpose(0, 1).contiguous()  # [B, n_head, C/n_head, T, F]

        x = x.view([batch_size, self.n_head * emb_dim, time, freq])  # [B, C, T, F]
        x = self.attn_concat_proj(x)  # [B, C, T, F]

        x = x + residual

        if self.dim == 4:
            x = x.transpose(-2, -1).contiguous()

        return x


class GlobalAttention(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int = None,
        ffn_name: str = "FeedForwardNetwork",
        kernel_size: int = 5,
        n_head: int = 8,
        dropout: float = 0.1,
        pos_enc: bool = True,
        *args,
        **kwargs,
    ):
        super(GlobalAttention, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan if hid_chan is not None else 2 * self.in_chan
        self.ffn_name = ffn_name
        self.kernel_size = kernel_size
        self.n_head = n_head
        self.dropout = dropout
        self.pos_enc = pos_enc

        self.MHSA = MultiHeadSelfAttention(self.in_chan, self.n_head, self.dropout, self.pos_enc)
        self.FFN = conv_layers.get(self.ffn_name)(self.in_chan, self.hid_chan, self.kernel_size, dropout=self.dropout)

    def forward(self, x: torch.Tensor):
        x = self.MHSA(x)
        x = self.FFN(x)
        return x


class GlobalAttention2D(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int = None,
        ffn_name: str = "FeedForwardNetwork",
        kernel_size: int = 5,
        n_head: int = 8,
        dropout: float = 0.1,
        single_ffn: bool = True,
        group_ffn: bool = False,
        pos_enc: bool = True,
        *args,
        **kwargs,
    ):
        super(GlobalAttention2D, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan if hid_chan is not None else 2 * self.in_chan
        self.ffn_name = ffn_name
        self.kernel_size = kernel_size
        self.n_head = n_head
        self.dropout = dropout
        self.single_ffn = single_ffn
        self.group_ffn = group_ffn
        self.pos_enc = pos_enc

        self.time_MHSA = MultiHeadSelfAttention(self.in_chan, self.n_head, self.dropout, self.pos_enc)
        self.freq_MHSA = MultiHeadSelfAttention(self.in_chan, self.n_head, self.dropout, self.pos_enc)

        self.group_FFN = nn.Identity()
        self.time_FFN = nn.Identity()
        self.freq_FFN = nn.Identity()

        if self.single_ffn:
            self.time_FFN = conv_layers.get(self.ffn_name)(self.in_chan, self.hid_chan, self.kernel_size, dropout=dropout)
            self.freq_FFN = conv_layers.get(self.ffn_name)(self.in_chan, self.hid_chan, self.kernel_size, dropout=dropout)

        if self.group_ffn:
            self.group_FFN = conv_layers.FeedForwardNetwork(self.in_chan, self.hid_chan, self.kernel_size, dropout=dropout, is2d=True)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.size()

        x = x.permute(0, 3, 1, 2).contiguous().view(B * W, C, H)
        x = self.time_MHSA.forward(x)
        x = self.time_FFN.forward(x)
        x = x.view(B, W, C, H).permute(0, 2, 3, 1).contiguous()

        x = self.group_FFN(x)

        x = x.permute(0, 2, 1, 3).contiguous().view(B * H, C, W)
        x = self.freq_MHSA.forward(x)
        x = self.freq_FFN.forward(x)
        x = x.view(B, H, C, W).permute(0, 2, 1, 3).contiguous()

        x = self.group_FFN(x)

        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_chan, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(in_chan, in_chan // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_chan // reduction, in_chan, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):
    def __init__(self, in_chan=512, reduction=16, kernel_size=49, *args, **kwargs):
        super().__init__()
        self.ca = ChannelAttention(in_chan=in_chan, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        residual = x
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x + residual


class ShuffleAttention(nn.Module):
    def __init__(self, in_chan=512, G=8, *args, **kwargs):
        super().__init__()
        self.G = G
        self.channel = in_chan
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(in_chan // (2 * G), in_chan // (2 * G))
        self.cweight = nn.Parameter(torch.zeros(1, in_chan // (2 * G), 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, in_chan // (2 * G), 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, in_chan // (2 * G), 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, in_chan // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x: torch.Tensor):
        b, _, h, w = x.size()
        # group into subfeatures
        x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w

        # channel_split
        x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w

        # channel attention
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
        x_channel = x_0 * self.sigmoid(x_channel)

        # spatial attention
        x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
        out = out.contiguous().view(b, -1, h, w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out


class CoTAttention(nn.Module):
    def __init__(self, in_chan=512, kernel_size=3, *args, **kwargs):
        super().__init__()
        self.dim = in_chan
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(in_chan, in_chan, kernel_size=kernel_size, padding=kernel_size // 2, groups=4, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.ReLU(),
        )
        self.value_embed = nn.Sequential(nn.Conv2d(in_chan, in_chan, 1, bias=False), nn.BatchNorm2d(in_chan))

        factor = 4
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * in_chan, 2 * in_chan // factor, 1, bias=False),
            nn.BatchNorm2d(2 * in_chan // factor),
            nn.ReLU(),
            nn.Conv2d(2 * in_chan // factor, kernel_size * kernel_size * in_chan, 1),
        )

        self.sm = nn.Softmax(-1)

    def forward(self, x: torch.Tensor):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)  # bs,c,h,w
        v = self.value_embed(x).view(bs, c, -1)  # bs,c,h,w

        y = torch.cat([k1, x], dim=1)  # bs,2c,h,w
        att = self.attention_embed(y)  # bs,c*k*k,h,w
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)  # bs,c,h*w
        k2 = self.sm(att) * v
        k2 = k2.view(bs, c, h, w)

        return k1 + k2
