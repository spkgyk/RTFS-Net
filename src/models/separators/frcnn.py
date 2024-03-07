import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import ConvNormAct


class FRCNNBlock(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        kernel_size: int = 5,
        stride: int = 2,
        norm_type: str = "gLN",
        act_type: str = "PReLU",
        upsampling_depth: int = 4,
        is2d: bool = False,
    ):
        super(FRCNNBlock, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_type = norm_type
        self.act_type = act_type
        self.upsampling_depth = upsampling_depth
        self.is2d = is2d

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
            is2d=self.is2d,
        )
        self.downsample_layers = self.__build_downsample_layers()
        self.fusion_layers = self.__build_fusion_layers()
        self.concat_layers = self.__build_concat_layers()
        self.residual_conv = nn.Sequential(
            ConvNormAct(
                self.hid_chan * self.upsampling_depth,
                self.hid_chan,
                1,
                norm_type=self.norm_type,
                act_type=self.act_type,
                is2d=self.is2d,
            ),
            ConvNormAct(
                self.hid_chan,
                self.in_chan,
                1,
                is2d=self.is2d,
            ),
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
        out = nn.ModuleList()
        for i in range(self.upsampling_depth):
            fuse_layer = nn.ModuleList()
            for j in range(self.upsampling_depth):
                if i == j or (j - i == 1):
                    fuse_layer.append(None)
                elif i - j == 1:
                    fuse_layer.append(
                        ConvNormAct(
                            in_chan=self.hid_chan,
                            out_chan=self.hid_chan,
                            kernel_size=self.kernel_size,
                            stride=self.stride,
                            groups=self.hid_chan,
                            norm_type=self.norm_type,
                            is2d=self.is2d,
                        )
                    )
            out.append(fuse_layer)
        return out

    def __build_concat_layers(self):
        out = nn.ModuleList()
        for i in range(self.upsampling_depth):
            if i == 0 or i == self.upsampling_depth - 1:
                out.append(
                    ConvNormAct(
                        in_chan=self.hid_chan * 2,
                        out_chan=self.hid_chan,
                        kernel_size=1,
                        norm_type=self.norm_type,
                        act_type=self.act_type,
                        is2d=self.is2d,
                    )
                )
            else:
                out.append(
                    ConvNormAct(
                        in_chan=self.hid_chan * 3,
                        out_chan=self.hid_chan,
                        kernel_size=1,
                        norm_type=self.norm_type,
                        act_type=self.act_type,
                        is2d=self.is2d,
                    )
                )
        return out

    def forward(self, x):
        # x: B, C, T, (F)
        residual = self.gateway(x)
        x_enc = self.projection(residual)

        # bottom-up
        downsampled_outputs = [self.downsample_layers[0](x_enc)]
        for i in range(1, self.upsampling_depth):
            downsampled_outputs.append(self.downsample_layers[i](downsampled_outputs[-1]))

        x_fused = []
        # lateral connection
        for i in range(self.upsampling_depth):
            shape = downsampled_outputs[i].shape
            y = torch.cat(
                (
                    self.fusion_layers[i][0](downsampled_outputs[i - 1]) if i - 1 >= 0 else torch.Tensor().to(x_enc.device),
                    downsampled_outputs[i],
                    F.interpolate(downsampled_outputs[i + 1], size=shape[-(len(shape) // 2) :], mode="nearest")
                    if i + 1 < self.upsampling_depth
                    else torch.Tensor().to(x_enc.device),
                ),
                dim=1,
            )
            x_fused.append(self.concat_layers[i](y))

        # resize to T
        shape = downsampled_outputs[0].shape
        for i in range(1, len(x_fused)):
            x_fused[i] = F.interpolate(x_fused[i], size=shape[-(len(shape) // 2) :], mode="nearest")

        out = self.residual_conv(torch.cat(x_fused, dim=1)) + residual

        return out


class FRCNN(nn.Module):
    def __init__(
        self,
        in_chan: int = -1,
        hid_chan: int = -1,
        kernel_size: int = 5,
        stride: int = 2,
        norm_type: str = "gLN",
        act_type: str = "PReLU",
        upsampling_depth: int = 4,
        repeats: int = 4,
        shared: bool = False,
        is2d: bool = False,
        *args,
        **kwargs,
    ):
        super(FRCNN, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_type = norm_type
        self.act_type = act_type
        self.upsampling_depth = upsampling_depth
        self.repeats = repeats
        self.shared = shared
        self.is2d = is2d

        self.blocks = self.__build_blocks()

    def __build_blocks(self):
        clss = FRCNNBlock if (self.in_chan > 0 and self.hid_chan > 0) else nn.Identity
        if self.shared:
            out = clss(
                in_chan=self.in_chan,
                hid_chan=self.hid_chan,
                kernel_size=self.kernel_size,
                stride=self.stride,
                norm_type=self.norm_type,
                act_type=self.act_type,
                upsampling_depth=self.upsampling_depth,
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
