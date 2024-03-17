import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import ConvNormAct, InjectionMultiSum, get


class TDANetBlock(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        kernel_size: int = 5,
        stride: int = 2,
        norm_type: str = "gLN",
        act_type: str = "PReLU",
        upsampling_depth: int = 4,
        layers: dict = dict(),
        is2d: bool = False,
    ):
        super(TDANetBlock, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_type = norm_type
        self.act_type = act_type
        self.upsampling_depth = upsampling_depth
        self.layers = layers
        self.is2d = is2d

        self.pool = F.adaptive_avg_pool2d if self.is2d else F.adaptive_avg_pool1d

        self.gateway = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.in_chan,
            kernel_size=1,
            groups=self.in_chan,
            act_type=self.act_type,
            is2d=self.is2d,
        )
        self.projection = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.hid_chan,
            kernel_size=1,
            norm_type=self.norm_type,
            act_type=self.act_type,
            is2d=self.is2d,
        )
        self.downsample_layers = self.__build_downsample_layers()
        self.globalatt = nn.Sequential(*[get(layer["layer_type"])(in_chan=self.hid_chan, **layer) for _, layer in self.layers.items()])
        self.fusion_layers = self.__build_fusion_layers()
        self.concat_layers = self.__build_concat_layers()
        self.residual_conv = ConvNormAct(
            in_chan=self.hid_chan,
            out_chan=self.in_chan,
            kernel_size=1,
            is2d=self.is2d,
        )

    def __build_downsample_layers(self):
        out = nn.ModuleList()
        for i in range(self.upsampling_depth):
            out.append(
                ConvNormAct(
                    in_chan=self.hid_chan,
                    out_chan=self.hid_chan,
                    kernel_size=self.kernel_size,
                    stride=1 if i == 0 else self.stride,
                    groups=self.hid_chan,
                    norm_type=self.norm_type,
                    is2d=self.is2d,
                )
            )

        return out

    def __build_fusion_layers(self):
        out = nn.ModuleList([])
        for _ in range(self.upsampling_depth):
            out.append(
                InjectionMultiSum(
                    in_chan=self.hid_chan,
                    kernel_size=self.kernel_size,
                    norm_type=self.norm_type,
                    is2d=self.is2d,
                )
            )

        return out

    def __build_concat_layers(self):
        out = nn.ModuleList([])
        for _ in range(self.upsampling_depth - 1):
            out.append(
                InjectionMultiSum(
                    in_chan=self.hid_chan,
                    kernel_size=self.kernel_size,
                    norm_type=self.norm_type,
                    is2d=self.is2d,
                )
            )

        return out

    def forward(self, x: torch.Tensor):
        # x: B, C, T, (F)
        residual = self.gateway(x)
        x_enc = self.projection(residual)

        # bottom-up
        downsampled_outputs = [self.downsample_layers[0](x_enc)]
        for i in range(1, self.upsampling_depth):
            downsampled_outputs.append(self.downsample_layers[i](downsampled_outputs[-1]))

        # global pooling
        shape = downsampled_outputs[-1].shape
        global_features = sum(self.pool(features, output_size=shape[-(len(shape) // 2) :]) for features in downsampled_outputs)

        # global attention module
        global_features = self.globalatt(global_features)  # B, N, T, (F)

        # Gather them now in reverse order
        x_fused = [self.fusion_layers[i](downsampled_outputs[i], global_features) for i in range(self.upsampling_depth)]

        # fuse them into a single vector
        expanded = self.concat_layers[-1](x_fused[-2], x_fused[-1]) + downsampled_outputs[-2]
        for i in range(self.upsampling_depth - 3, -1, -1):
            expanded = self.concat_layers[i](x_fused[i], expanded) + downsampled_outputs[i]

        out = self.residual_conv(expanded) + residual

        return out


class TDANet(nn.Module):
    def __init__(
        self,
        in_chan: int = -1,
        hid_chan: int = -1,
        kernel_size: int = 5,
        stride: int = 2,
        norm_type: str = "gLN",
        act_type: str = "PReLU",
        upsampling_depth: int = 4,
        layers: dict = dict(),
        repeats: int = 4,
        shared: bool = False,
        is2d: bool = False,
        *args,
        **kwargs,
    ):
        super(TDANet, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_type = norm_type
        self.act_type = act_type
        self.upsampling_depth = upsampling_depth
        self.layers = layers
        self.repeats = repeats
        self.shared = shared
        self.is2d = is2d

        self.blocks = self.__build_blocks()

    def __build_blocks(self):
        clss = TDANetBlock if (self.in_chan > 0 and self.hid_chan > 0) else nn.Identity
        if self.shared:
            out = clss(
                in_chan=self.in_chan,
                hid_chan=self.hid_chan,
                kernel_size=self.kernel_size,
                stride=self.stride,
                norm_type=self.norm_type,
                act_type=self.act_type,
                upsampling_depth=self.upsampling_depth,
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
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        norm_type=self.norm_type,
                        act_type=self.act_type,
                        upsampling_depth=self.upsampling_depth,
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
