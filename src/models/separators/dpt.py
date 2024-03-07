import torch
import torch.nn as nn

from ..layers import ConvNormAct, get


class DPTNetBlock(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        layers: dict = dict(),
        is2d: bool = False,
    ):
        super(DPTNetBlock, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.layers = layers
        self.is2d = is2d

        self.gateway = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.in_chan,
            kernel_size=1,
            groups=self.in_chan,
            act_type="PReLU",
            is2d=self.is2d,
        )
        self.projection = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.hid_chan,
            kernel_size=1,
            is2d=self.is2d,
        )
        self.globalatt = nn.Sequential(*[get(layer["layer_type"])(in_chan=self.hid_chan, **layer) for _, layer in self.layers.items()])
        self.residual_conv = ConvNormAct(
            in_chan=self.hid_chan,
            out_chan=self.in_chan,
            kernel_size=1,
            is2d=self.is2d,
        )

    def forward(self, x: torch.Tensor):
        residual = self.gateway(x)
        x = self.projection(residual)
        x = self.globalatt(x)
        x = self.residual_conv(x) + residual
        return x


class DPTNet(nn.Module):
    def __init__(
        self,
        in_chan: int = -1,
        hid_chan: int = -1,
        layers: list[dict] = [],
        repeats: int = 4,
        shared: bool = False,
        is2d: bool = False,
        *args,
        **kwargs,
    ):
        super(DPTNet, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.layers = layers
        self.repeats = repeats
        self.shared = shared
        self.is2d = is2d

        self.blocks = self.__build_blocks()

    def __build_blocks(self):
        clss = DPTNetBlock if self.in_chan > 0 else nn.Identity
        if self.shared:
            out = clss(
                in_chan=self.in_chan,
                hid_chan=self.hid_chan,
                layers=self.layers,
                is2d=self.is2d,
            )
        else:
            out = nn.ModuleList()
            for _ in range(self.repeats):
                out.append(
                    clss(
                        in_chan=self.in_chan,
                        hid_chan=self.hid_chan,
                        layers=self.layers,
                        is2d=self.is2d,
                    )
                )

        return out

    def get_block(self, i: int):
        if self.shared:
            return self.blocks
        else:
            return self.blocks[i]

    def forward(self, x: torch.Tensor):
        residual = x
        for i in range(self.repeats):
            x = self.get_block(i)((x + residual) if i > 0 else x)
        return x
