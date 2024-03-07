import torch
import torch.nn as nn

from thop import profile

from .shufflenetv2 import ShuffleNetV2
from .resnet import ResNet, BasicBlock


def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch * s_time, n_channels, sx, sy)


class FRCNNVideoModel(nn.Module):
    def __init__(
        self,
        backbone_type="resnet",
        relu_type="prelu",
        width_mult=1.0,
        pretrain=None,
        print_macs=True,
        *args,
        **kwargs,
    ):
        super(FRCNNVideoModel, self).__init__()
        self.backbone_type = backbone_type
        if self.backbone_type == "resnet":
            self.frontend_nout = 64
            self.backend_out = 512
            self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)
        elif self.backbone_type == "shufflenet":
            assert width_mult in [0.5, 1.0, 1.5, 2.0], "Width multiplier not correct"
            shufflenet = ShuffleNetV2(input_size=96, width_mult=width_mult)
            self.trunk = nn.Sequential(shufflenet.features, shufflenet.conv_last, shufflenet.globalpool)
            self.frontend_nout = 24
            self.backend_out = 1024 if width_mult != 2.0 else 2048
            self.stage_out_channels = shufflenet.stage_out_channels[-1]

        frontend_relu = nn.PReLU(num_parameters=self.frontend_nout) if relu_type == "prelu" else nn.ReLU()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                1,
                self.frontend_nout,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.pretrain = pretrain
        if pretrain:
            self.init_from(pretrain)

        if print_macs:
            self.get_MACs()

    def forward(self, x: torch.Tensor):
        B, C, T, H, W = x.size()
        x = self.frontend3D(x)
        Tnew = x.shape[2]  # outpu should be B x C2 x Tnew x H x W
        x = threeD_to_2D_tensor(x)
        x = self.trunk(x)
        if self.backbone_type == "shufflenet":
            x = x.view(-1, self.stage_out_channels)
        x = x.view(B, Tnew, x.size(1)).transpose(1, 2).contiguous()

        return x

    def init_from(self, path):
        pretrained_dict = torch.load(path, map_location="cpu")["model_state_dict"]
        update_frcnn_parameter(self, pretrained_dict)

    def train(self, mode=True):
        super().train(mode)
        if mode:  # freeze BN stats
            for m in self.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()

    def get_MACs(self):
        with torch.no_grad():
            batch_size = 1
            seconds = 2
            h, w = 88, 88
            device = next(self.parameters()).device
            video_input = torch.rand(batch_size, 1, seconds * 25, h, w).to(device)

            self.macs = profile(self, inputs=(video_input,), verbose=False)[0] / 1000000
            self.number_of_parameters = sum(p.numel() for p in self.parameters()) / 1000

            s = "Pretrained Video Backbone\nNumber of MACs: {:,.1f}M\nNumber of parameters: {:,.1f}K\n".format(
                self.macs, self.number_of_parameters
            )

            print(s)


def update_frcnn_parameter(model, pretrained_dict):
    model_dict = model.state_dict()
    update_dict = {}
    for k, v in pretrained_dict.items():
        if "tcn" in k:
            pass
        else:
            update_dict[k] = v
    model_dict.update(update_dict)
    model.load_state_dict(model_dict)
    for p in model.parameters():
        p.requires_grad = False
    return model
