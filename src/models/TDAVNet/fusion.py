import torch
import torch.nn as nn
import torch.nn.functional as F


from ..layers import ConvNormAct, InjectionMultiSum, ConvLSTMFusionCell, ConvGRUFusionCell, ATTNFusionCell


class FusionBasemodule(nn.Module):
    def __init__(self, ain_chan: int, vin_chan: int, kernel_size: int, video_fusion: bool, is2d: bool):
        super(FusionBasemodule, self).__init__()
        self.ain_chan = ain_chan
        self.vin_chan = vin_chan
        self.kernel_size = kernel_size
        self.video_fusion = video_fusion
        self.is2d = is2d

    def forward(self, audio, video):
        raise NotImplementedError

    def wrangle_dims(self, audio: torch.Tensor, video: torch.Tensor):
        T1 = audio.shape[-(len(audio.shape) // 2) :]
        T2 = video.shape[-(len(video.shape) // 2) :]

        self.x = len(T1) > len(T2)
        self.y = len(T2) > len(T1)

        video = video.unsqueeze(-1) if self.x else video
        audio = audio.unsqueeze(-1) if self.y else audio

        return audio, video

    def unwrangle_dims(self, audio: torch.Tensor, video: torch.Tensor):
        video = video.squeeze(-1) if self.x else video
        audio = audio.squeeze(-1) if self.y else audio

        return audio, video


class ConcatFusion(FusionBasemodule):
    def __init__(self, ain_chan: int, vin_chan: int, kernel_size: int, video_fusion: bool = True, is2d: bool = False):
        super(ConcatFusion, self).__init__(ain_chan, vin_chan, kernel_size, video_fusion, is2d)

        self.audio_conv = ConvNormAct(self.ain_chan + self.vin_chan, self.ain_chan, self.kernel_size, norm_type="gLN", is2d=self.is2d)
        if video_fusion:
            self.video_conv = ConvNormAct(self.ain_chan + self.vin_chan, self.vin_chan, self.kernel_size, norm_type="gLN", is2d=self.is2d)

    def forward(self, audio: torch.Tensor, video: torch.Tensor):
        audio, video = self.wrangle_dims(audio, video)

        video_interp = F.interpolate(video, size=audio.shape[-(len(audio.shape) // 2) :], mode="nearest")
        audio_video_concat = torch.cat([audio, video_interp], dim=1)
        audio_fused = self.audio_conv(audio_video_concat)

        if self.video_fusion:
            audio_interp = F.interpolate(audio, size=video.shape[-(len(video.shape) // 2) :], mode="nearest")
            video_audio_concat = torch.cat([audio_interp, video], dim=1)
            video_fused = self.video_conv(video_audio_concat)
        else:
            video_fused = video

        audio_fused, video_fused = self.unwrangle_dims(audio_fused, video_fused)

        return audio_fused, video_fused


class SumFusion(FusionBasemodule):
    def __init__(self, ain_chan: int, vin_chan: int, kernel_size: int, video_fusion: bool = True, is2d: bool = False):
        super(SumFusion, self).__init__(ain_chan, vin_chan, kernel_size, video_fusion, is2d)

        if video_fusion:
            self.audio_conv = ConvNormAct(self.ain_chan, self.vin_chan, self.kernel_size, norm_type="gLN", is2d=self.is2d)
        self.video_conv = ConvNormAct(self.vin_chan, self.ain_chan, self.kernel_size, norm_type="gLN", is2d=self.is2d)

    def forward(self, audio: torch.Tensor, video: torch.Tensor):
        audio, video = self.wrangle_dims(audio, video)

        if self.video_fusion:
            audio_interp = F.interpolate(audio, size=video.shape[-(len(video.shape) // 2) :], mode="nearest")
            video_fused = self.audio_conv(audio_interp) + video
        else:
            video_fused = video

        video_interp = F.interpolate(video, size=audio.shape[-(len(audio.shape) // 2) :], mode="nearest")
        audio_fused = self.video_conv(video_interp) + audio

        audio_fused, video_fused = self.unwrangle_dims(audio_fused, video_fused)

        return audio_fused, video_fused


class InjectionFusion(FusionBasemodule):
    def __init__(self, ain_chan: int, vin_chan: int, kernel_size: int, video_fusion: bool = True, is2d: bool = False):
        super(InjectionFusion, self).__init__(ain_chan, vin_chan, kernel_size, video_fusion, is2d)

        if video_fusion:
            self.audio_conv = ConvNormAct(self.ain_chan, self.vin_chan, 1, is2d=self.is2d)
            self.video_inj = InjectionMultiSum(self.vin_chan, self.kernel_size, "gLN", is2d=self.is2d)
        self.video_conv = ConvNormAct(self.vin_chan, self.ain_chan, 1, is2d=self.is2d)
        self.audio_inj = InjectionMultiSum(self.ain_chan, self.kernel_size, "gLN", is2d=self.is2d)

    def forward(self, audio: torch.Tensor, video: torch.Tensor):
        audio, video = self.wrangle_dims(audio, video)

        if self.video_fusion:
            video_fused = self.video_inj(video, self.audio_conv(audio))
        else:
            video_fused = video

        audio_fused = self.audio_inj(audio, self.video_conv(video))

        audio_fused, video_fused = self.unwrangle_dims(audio_fused, video_fused)

        return audio_fused, video_fused


class LSTMFusion(FusionBasemodule):
    def __init__(
        self,
        ain_chan: int,
        vin_chan: int,
        kernel_size: int,
        video_fusion: bool = True,
        is2d=True,
        bidirectional: bool = True,
        *args,
        **kwargs,
    ):
        super(LSTMFusion, self).__init__(ain_chan, vin_chan, kernel_size, video_fusion, is2d)

        self.bidirectional = bidirectional

        if video_fusion:
            self.video_lstm = ConvLSTMFusionCell(self.vin_chan, self.ain_chan, self.kernel_size, self.bidirectional, self.is2d)
        self.audio_lstm = ConvLSTMFusionCell(self.ain_chan, self.vin_chan, self.kernel_size, self.bidirectional, self.is2d)

    def forward(self, audio: torch.Tensor, video: torch.Tensor):
        audio, video = self.wrangle_dims(audio, video)

        if self.video_fusion:
            video_fused = self.video_lstm(video, audio)
        else:
            video_fused = video

        audio_fused = self.audio_lstm(audio, video)

        audio_fused, video_fused = self.unwrangle_dims(audio_fused, video_fused)

        return audio_fused, video_fused


class GRUFusion(FusionBasemodule):
    def __init__(
        self,
        ain_chan: int,
        vin_chan: int,
        kernel_size: int,
        video_fusion: bool = True,
        is2d=True,
        bidirectional: bool = True,
        *args,
        **kwargs,
    ):
        super(GRUFusion, self).__init__(ain_chan, vin_chan, kernel_size, video_fusion, is2d)

        self.bidirectional = bidirectional

        if video_fusion:
            self.video_lstm = ConvGRUFusionCell(self.vin_chan, self.ain_chan, self.kernel_size, self.bidirectional, self.is2d)
        self.audio_lstm = ConvGRUFusionCell(self.ain_chan, self.vin_chan, self.kernel_size, self.bidirectional, self.is2d)

    def forward(self, audio: torch.Tensor, video: torch.Tensor):
        audio, video = self.wrangle_dims(audio, video)

        if self.video_fusion:
            video_fused = self.video_lstm(video, audio)
        else:
            video_fused = video

        audio_fused = self.audio_lstm(audio, video)

        audio_fused, video_fused = self.unwrangle_dims(audio_fused, video_fused)

        return audio_fused, video_fused


class ATTNFusion(FusionBasemodule):
    def __init__(
        self,
        ain_chan: int,
        vin_chan: int,
        kernel_size: int,
        video_fusion: bool = True,
        is2d=True,
        *args,
        **kwargs,
    ):
        super(ATTNFusion, self).__init__(ain_chan, vin_chan, kernel_size, video_fusion, is2d)

        if video_fusion:
            self.video_lstm = ATTNFusionCell(self.vin_chan, self.ain_chan, self.kernel_size, self.is2d)
        self.audio_lstm = ATTNFusionCell(self.ain_chan, self.vin_chan, self.kernel_size, self.is2d)

    def forward(self, audio: torch.Tensor, video: torch.Tensor):
        if self.video_fusion:
            video_fused = self.video_lstm(video, audio)
        else:
            video_fused = video

        audio_fused = self.audio_lstm(audio, video)

        return audio_fused, video_fused


class MultiModalFusion(nn.Module):
    def __init__(
        self,
        audio_bn_chan: int,
        video_bn_chan: int,
        kernel_size: int = 1,
        fusion_repeats: int = 3,
        fusion_type: str = "ConcatFusion",
        fusion_shared: bool = False,
        is2d: bool = False,
        **kwargs,
    ):
        super(MultiModalFusion, self).__init__()
        self.audio_bn_chan = audio_bn_chan
        self.video_bn_chan = video_bn_chan
        self.kernel_size = kernel_size
        self.fusion_repeats = fusion_repeats
        self.fusion_type = fusion_type
        self.fusion_shared = fusion_shared
        self.is2d = is2d

        self.fusion_module = self.__build_fusion_module(**kwargs)

    def __build_fusion_module(self, **kwargs):
        fusion_class = globals().get(self.fusion_type) if self.fusion_repeats > 0 else nn.Identity
        if self.fusion_shared:
            out = fusion_class(
                ain_chan=self.audio_bn_chan,
                vin_chan=self.video_bn_chan,
                kernel_size=self.kernel_size,
                video_fusion=self.fusion_repeats > 1,
                is2d=self.is2d,
                **kwargs,
            )
        else:
            out = nn.ModuleList()
            for i in range(self.fusion_repeats):
                out.append(
                    fusion_class(
                        ain_chan=self.audio_bn_chan,
                        vin_chan=self.video_bn_chan,
                        kernel_size=self.kernel_size,
                        video_fusion=i != self.fusion_repeats - 1,
                        is2d=self.is2d,
                        **kwargs,
                    )
                )

        return out

    def get_fusion_block(self, i: int):
        if self.fusion_shared:
            return self.fusion_module
        else:
            return self.fusion_module[i]

    def forward(self, audio: torch.Tensor, video: torch.Tensor):
        audio_residual = audio
        video_residual = video

        for i in range(self.fusion_repeats):
            if i == 0:
                audio_fused, video_fused = self.get_fusion_block(i)(audio, video)
            else:
                audio_fused, video_fused = self.get_fusion_block(i)(audio_fused + audio_residual, video_fused + video_residual)

        return audio_fused
