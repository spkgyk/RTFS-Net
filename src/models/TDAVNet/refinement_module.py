import inspect
import torch
import torch.nn as nn

from .fusion import MultiModalFusion
from ..utils import get_MACS_params
from .. import separators


class RefinementModule(nn.Module):
    def __init__(
        self,
        audio_params: dict,
        video_params: dict,
        audio_bn_chan: int,
        video_bn_chan: int,
        fusion_params: dict,
    ):
        super(RefinementModule, self).__init__()
        self.audio_params = audio_params
        self.video_params = video_params
        self.audio_bn_chan = audio_bn_chan
        self.video_bn_chan = video_bn_chan
        self.fusion_params = fusion_params

        self.fusion_repeats = self.video_params.get("repeats", 0)
        self.audio_repeats = self.audio_params["repeats"] - self.fusion_repeats

        self.audio_net = separators.get(self.audio_params.get("audio_net", None))(
            **self.audio_params,
            in_chan=self.audio_bn_chan,
        )
        self.video_net = separators.get(self.video_params.get("video_net", None))(
            **self.video_params,
            in_chan=self.video_bn_chan,
        )

        self.crossmodal_fusion = MultiModalFusion(
            **self.fusion_params,
            audio_bn_chan=self.audio_bn_chan,
            video_bn_chan=self.video_bn_chan,
            fusion_repeats=self.fusion_repeats,
        )

    def forward(self, audio: torch.Tensor, video: torch.Tensor):
        audio_residual = audio
        video_residual = video

        # cross modal fusion
        for i in range(self.fusion_repeats):
            audio = self.audio_net.get_block(i)(audio + audio_residual if i > 0 else audio)
            video = self.video_net.get_block(i)(video + video_residual if i > 0 else video)

            audio, video = self.crossmodal_fusion.get_fusion_block(i)(audio, video)

        # further refinement
        for j in range(self.audio_repeats):
            i = j + self.fusion_repeats

            audio = self.audio_net.get_block(i)(audio + audio_residual if i > 0 else audio)

        return audio

    def get_config(self):
        encoder_args = {}

        for k, v in (self.__dict__).items():
            if not k.startswith("_") and k != "training":
                if not inspect.ismethod(v):
                    encoder_args[k] = v

        return encoder_args

    def get_MACs(self, bn_audio, bn_video):
        macs = []

        macs += get_MACS_params(self.audio_net, (bn_audio,))

        macs += get_MACS_params(self.video_net, (bn_video,))

        macs += get_MACS_params(self.crossmodal_fusion, (bn_audio, bn_video))

        return macs
