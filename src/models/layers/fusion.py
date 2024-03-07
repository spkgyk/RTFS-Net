import torch
import torch.nn as nn
import torch.nn.functional as F


from .conv_layers import ConvNormAct


class InjectionMultiSum(nn.Module):
    def __init__(
        self,
        in_chan: int,
        kernel_size: int,
        norm_type: str = "gLN",
        is2d: bool = False,
        *args,
        **kwargs,
    ):
        super(InjectionMultiSum, self).__init__()
        self.in_chan = in_chan
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.is2d = is2d

        self.local_embedding = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.in_chan,
            kernel_size=self.kernel_size,
            groups=self.in_chan,
            norm_type=self.norm_type,
            bias=False,
            is2d=self.is2d,
        )
        self.global_embedding = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.in_chan,
            kernel_size=self.kernel_size,
            groups=self.in_chan,
            norm_type=self.norm_type,
            bias=False,
            is2d=self.is2d,
        )
        self.global_gate = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.in_chan,
            kernel_size=self.kernel_size,
            groups=self.in_chan,
            norm_type=self.norm_type,
            act_type="Sigmoid",
            bias=False,
            is2d=self.is2d,
        )

    def forward(self, local_features: torch.Tensor, global_features: torch.Tensor):
        old_shape = global_features.shape[-(len(local_features.shape) // 2) :]
        new_shape = local_features.shape[-(len(local_features.shape) // 2) :]

        local_emb = self.local_embedding(local_features)
        if torch.prod(torch.tensor(new_shape)) > torch.prod(torch.tensor(old_shape)):
            global_emb = F.interpolate(self.global_embedding(global_features), size=new_shape, mode="nearest")
            gate = F.interpolate(self.global_gate(global_features), size=new_shape, mode="nearest")
        else:
            g_interp = F.interpolate(global_features, size=new_shape, mode="nearest")
            global_emb = self.global_embedding(g_interp)
            gate = self.global_gate(g_interp)

        injection_sum = local_emb * gate + global_emb

        return injection_sum


class ConvLSTMFusionCell(nn.Module):
    def __init__(self, in_chan_a: int, in_chan_b, kernel_size: int = 1, bidirectional: bool = False, is2d: bool = False, *args, **kwargs):
        super(ConvLSTMFusionCell, self).__init__()
        self.in_chan_a = in_chan_a
        self.in_chan_b = in_chan_b
        self.kernel_size = kernel_size
        self.bidirectional = bidirectional
        self.num_dir = int(bidirectional) + 1
        self.is2d = is2d

        self.conv_a = ConvNormAct(
            self.in_chan_a * self.num_dir,
            self.in_chan_a * 4,
            self.kernel_size,
            is2d=self.is2d,
            groups=self.in_chan_a // 4,
            norm_type="gLN",
        )
        self.conv_b = ConvNormAct(
            self.in_chan_b * self.num_dir,
            self.in_chan_a * 4,
            self.kernel_size,
            is2d=self.is2d,
            groups=self.in_chan_a // 4,
            norm_type="gLN",
        )

    def forward(self, tensor_a: torch.Tensor, tensor_b: torch.Tensor):
        if self.bidirectional:
            a_flipped = tensor_a.flip(-1).flip(-2) if self.is2d else tensor_a.flip(-1)
            b_flipped = tensor_b.flip(-1).flip(-2) if self.is2d else tensor_b.flip(-1)
            tensor_a = torch.cat((tensor_a, a_flipped), dim=1)
            tensor_b = torch.cat((tensor_b, b_flipped), dim=1)

        old_shape = tensor_b.shape[-(len(tensor_a.shape) // 2) :]
        new_shape = tensor_a.shape[-(len(tensor_a.shape) // 2) :]

        if torch.prod(torch.tensor(new_shape)) > torch.prod(torch.tensor(old_shape)):
            gates = self.conv_a(tensor_a) + F.interpolate(self.conv_b(tensor_b), size=new_shape, mode="nearest")
        else:
            gates = self.conv_a(tensor_a) + self.conv_b(F.interpolate(tensor_b, size=new_shape, mode="nearest"))

        i_t, f_t, g_t, o_t = gates.chunk(4, 1)

        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        g_t = torch.tanh(g_t)
        o_t = torch.sigmoid(o_t)

        c_next = f_t + (i_t * g_t)
        h_next = o_t * torch.tanh(c_next)

        return h_next


class ConvGRUFusionCell(nn.Module):
    def __init__(
        self,
        in_chan_a: int,
        in_chan_b: int,
        kernel_size: int = 1,
        bidirectional: bool = False,
        is2d: bool = False,
        *args,
        **kwargs,
    ):
        super(ConvGRUFusionCell, self).__init__()
        self.in_chan_a = in_chan_a
        self.in_chan_b = in_chan_b
        self.kernel_size = kernel_size
        self.bidirectional = bidirectional
        self.num_dir = int(bidirectional) + 1
        self.is2d = is2d

        # For GRU, we have 3 gates: reset, update, and new
        self.conv_a = ConvNormAct(
            self.in_chan_a * self.num_dir,
            self.in_chan_a * 3,  # times 3 for the three GRU gates
            self.kernel_size,
            is2d=self.is2d,
            groups=self.in_chan_a,
            norm_type="gLN",
        )
        self.conv_b = ConvNormAct(
            self.in_chan_b * self.num_dir,
            self.in_chan_a * 3,  # times 3 for the three GRU gates
            self.kernel_size,
            is2d=self.is2d,
            groups=self.in_chan_a,
            norm_type="gLN",
        )

    def forward(self, tensor_a: torch.Tensor, tensor_b: torch.Tensor):
        if self.bidirectional:
            a_flipped = tensor_a.flip(-1).flip(-2) if self.is2d else tensor_a.flip(-1)
            b_flipped = tensor_b.flip(-1).flip(-2) if self.is2d else tensor_b.flip(-1)
            tensor_a = torch.cat((tensor_a, a_flipped), dim=1)
            tensor_b = torch.cat((tensor_b, b_flipped), dim=1)

        old_shape = tensor_b.shape[-(len(tensor_a.shape) // 2) :]
        new_shape = tensor_a.shape[-(len(tensor_a.shape) // 2) :]

        x = self.conv_a(tensor_a)
        if torch.prod(torch.tensor(new_shape)) > torch.prod(torch.tensor(old_shape)):
            h = F.interpolate(self.conv_b(tensor_b), size=new_shape, mode="nearest")
        else:
            h = self.conv_b(F.interpolate(tensor_b, size=new_shape, mode="nearest"))

        # Split the gates: r_t (reset), z_t (update), n_t (new)
        x_r, x_z, x_n = x.chunk(3, 1)
        h_r, h_z, h_n = h.chunk(3, 1)

        r_t = torch.sigmoid(x_r + h_r)
        z_t = torch.sigmoid(x_z + h_z)
        n_t = torch.tanh(x_n + r_t * h_n)

        # GRU Logic
        h_next = (1 - z_t) * n_t

        return h_next


class ATTNFusionCell(nn.Module):
    def __init__(
        self,
        in_chan_a: int,
        in_chan_b: int,
        kernel_size: int = 1,
        is2d: bool = False,
        *args,
        **kwargs,
    ):
        super(ATTNFusionCell, self).__init__()
        self.in_chan_a = in_chan_a
        self.in_chan_b = in_chan_b
        self.kernel_size = kernel_size
        self.is2d = is2d

        self.key_embed = ConvNormAct(
            self.in_chan_a,
            self.in_chan_a,
            1,
            groups=self.in_chan_a,
            norm_type="BatchNorm2d",
            act_type="ReLU",
            bias=False,
            is2d=self.is2d,
        )
        self.value_embed = ConvNormAct(
            self.in_chan_a,
            self.in_chan_a,
            1,
            groups=self.in_chan_a,
            norm_type="BatchNorm2d",
            bias=False,
            is2d=self.is2d,
        )
        self.attention_embed = ConvNormAct(
            self.in_chan_b,
            self.kernel_size * self.in_chan_a,
            1,
            groups=self.in_chan_a,
            norm_type="gLN",
        )

        self.resize = ConvNormAct(
            self.in_chan_b,
            self.in_chan_a,
            1,
            groups=self.in_chan_a,
            norm_type="gLN",
        )

        # print(
        #     int(sum(p.numel() for p in self.resize.parameters() if p.requires_grad) / 1000),
        #     int(sum(p.numel() for p in self.key_embed.parameters() if p.requires_grad) / 1000),
        #     int(sum(p.numel() for p in self.value_embed.parameters() if p.requires_grad) / 1000),
        #     int(sum(p.numel() for p in self.attention_embed.parameters() if p.requires_grad) / 1000),
        # )

    def forward(self, tensor_a: torch.Tensor, tensor_b: torch.Tensor):
        batch_size, _, time_steps, _ = tensor_a.shape

        b_transformed = F.interpolate(self.resize(tensor_b), size=time_steps, mode="nearest")
        if self.is2d:
            b_transformed = b_transformed.unsqueeze(-1)

        k1 = self.key_embed(tensor_a) * b_transformed  # bs,c,h,w
        v = self.value_embed(tensor_a)  # bs,c,h,w

        att = self.attention_embed(tensor_b)  # bs,c*k*k,h,w
        att = att.reshape(batch_size, self.in_chan_a, self.kernel_size, -1)
        att = att.mean(2, keepdim=False).view(batch_size, self.in_chan_a, -1)  # bs,c,h*w
        att = F.interpolate(torch.softmax(att, -1), size=time_steps, mode="nearest")

        if self.is2d:
            att = att.unsqueeze(-1)
        k2 = att * v

        # Fusion
        fused_tensor = k1 + k2

        return fused_tensor
