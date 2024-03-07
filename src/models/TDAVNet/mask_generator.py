import torch
import inspect
import torch.nn as nn

from ..layers import ConvNormAct, activations


class BaseMaskGenerator(nn.Module):
    def get_config(self):
        encoder_args = {}

        for k, v in (self.__dict__).items():
            if not k.startswith("_") and k != "training":
                if not inspect.ismethod(v):
                    encoder_args[k] = v

        return encoder_args


class MaskGenerator(BaseMaskGenerator):
    def __init__(
        self,
        n_src: int,
        audio_emb_dim: int,
        bottleneck_chan: int,
        kernel_size: int = 1,
        mask_act: str = "ReLU",
        RI_split: bool = False,
        output_gate: bool = False,
        dw_gate: bool = False,
        direct: bool = False,
        is2d: bool = False,
        *args,
        **kwargs,
    ):
        super(MaskGenerator, self).__init__()
        self.n_src = n_src
        self.in_chan = audio_emb_dim
        self.bottleneck_chan = bottleneck_chan
        self.kernel_size = kernel_size
        self.mask_act = mask_act
        self.output_gate = output_gate
        self.dw_gate = dw_gate
        self.RI_split = RI_split
        self.direct = direct
        self.is2d = is2d

        if not self.direct:
            mask_output_chan = self.n_src * self.in_chan

            self.mask_generator = nn.Sequential(
                nn.PReLU(),
                ConvNormAct(
                    self.bottleneck_chan,
                    mask_output_chan,
                    self.kernel_size,
                    act_type=self.mask_act,
                    is2d=self.is2d,
                ),
            )

            if self.output_gate:
                groups = mask_output_chan if self.dw_gate else 1
                self.output = ConvNormAct(mask_output_chan, mask_output_chan, 1, act_type="Tanh", is2d=self.is2d, groups=groups)
                self.gate = ConvNormAct(mask_output_chan, mask_output_chan, 1, act_type="Sigmoid", is2d=self.is2d, groups=groups)

    def __apply_masks(self, masks: torch.Tensor, audio_mixture_embedding: torch.Tensor):
        batch_size = audio_mixture_embedding.size(0)
        dims = audio_mixture_embedding.shape[-(len(audio_mixture_embedding.shape) // 2) :]
        if self.RI_split:
            masks = masks.view(batch_size, self.n_src, 2, self.in_chan // 2, *dims)
            audio_mixture_embedding = audio_mixture_embedding.view(batch_size, 2, self.in_chan // 2, *dims)

            mask_real = masks[:, :, 0]  # B, n_src, C/2, T, (F)
            mask_imag = masks[:, :, 1]  # B, n_src, C/2, T, (F)
            emb_real = audio_mixture_embedding[:, 0].unsqueeze(1)  # B, 1, C/2, T, (F)
            emb_imag = audio_mixture_embedding[:, 1].unsqueeze(1)  # B, 1, C/2, T, (F)

            est_spec_real = emb_real * mask_real - emb_imag * mask_imag  # B, n_src, C/2, T, (F)
            est_spec_imag = emb_real * mask_imag + emb_imag * mask_real  # B, n_src, C/2, T, (F)

            separated_audio_embedding = torch.cat([est_spec_real, est_spec_imag], 2)  # B, n_src, C, T, (F)
        else:
            masks = masks.view(batch_size, self.n_src, self.in_chan, *dims)
            separated_audio_embedding = masks * audio_mixture_embedding.unsqueeze(1)

        return separated_audio_embedding

    def forward(self, refined_features: torch.Tensor, audio_mixture_embedding: torch.Tensor):
        if self.direct:
            return refined_features
        else:
            masks = self.mask_generator(refined_features)
            if self.output_gate:
                masks = self.output(masks) * self.gate(masks)

            separated_audio_embedding = self.__apply_masks(masks, audio_mixture_embedding)

            return separated_audio_embedding


class MaskGenerator2Chan(BaseMaskGenerator):
    def __init__(
        self,
        n_src: int,
        bottleneck_chan: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = False,
        mask_act: str = "ReLU",
        RI_split: bool = False,
        output_gate: bool = False,
        dw_gate: bool = False,
        direct: bool = False,
        *args,
        **kwargs,
    ):
        super(MaskGenerator2Chan, self).__init__()
        self.n_src = n_src
        self.bottleneck_chan = bottleneck_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.mask_act = mask_act
        self.output_gate = output_gate
        self.dw_gate = dw_gate
        self.RI_split = RI_split
        self.direct = direct

        mask_output_chan = self.n_src * 2

        self.mask_generator = nn.Sequential(
            nn.PReLU(),
            nn.ConvTranspose2d(
                self.bottleneck_chan,
                mask_output_chan,
                self.kernel_size,
                self.stride,
                padding=(self.kernel_size - 1) // 2,
                bias=self.bias,
            ),
            activations.get(self.mask_act)(),
        )

        if self.output_gate:
            groups = mask_output_chan if self.dw_gate else 1
            self.output = ConvNormAct(mask_output_chan, mask_output_chan, 1, act_type="Tanh", is2d=True, groups=groups)
            self.gate = ConvNormAct(mask_output_chan, mask_output_chan, 1, act_type="Sigmoid", is2d=True, groups=groups)

    def __apply_masks(self, masks: torch.Tensor, audio_mixture_embedding: torch.Tensor):
        batch_size = audio_mixture_embedding.size(0)
        dims = audio_mixture_embedding.shape[-(len(audio_mixture_embedding.shape) // 2) :]
        if self.RI_split:
            masks = masks.view(batch_size, self.n_src, 2, 1, *dims)
            audio_mixture_embedding = audio_mixture_embedding.view(batch_size, 2, 1, *dims)

            mask_real = masks[:, :, 0]  # B, n_src, 1, T, (F)
            mask_imag = masks[:, :, 1]  # B, n_src, 1, T, (F)
            emb_real = audio_mixture_embedding[:, 0].unsqueeze(1)  # B, 1, 1, T, (F)
            emb_imag = audio_mixture_embedding[:, 1].unsqueeze(1)  # B, 1, 1, T, (F)

            est_spec_real = emb_real * mask_real - emb_imag * mask_imag  # B, n_src, 1, T, (F)
            est_spec_imag = emb_real * mask_imag + emb_imag * mask_real  # B, n_src, 1, T, (F)

            separated_audio_embedding = torch.cat([est_spec_real, est_spec_imag], 2)  # B, n_src, 2, T, (F)
        else:
            masks = masks.view(batch_size, self.n_src, 2, *dims)
            separated_audio_embedding = masks * audio_mixture_embedding.unsqueeze(1)

        return separated_audio_embedding

    def forward(self, refined_features: torch.Tensor, audio_mixture_embedding: torch.Tensor):
        if self.direct:
            x = refined_features.shape
            refined_features = self.mask_generator(refined_features)
            if self.output_gate:
                refined_features = self.output(refined_features) * self.gate(refined_features)

            return refined_features.view(x[0], self.n_src, 2, *x[2:])
        else:
            masks = self.mask_generator(refined_features)
            if self.output_gate:
                masks = self.output(masks) * self.gate(masks)

            separated_audio_embedding = self.__apply_masks(masks, audio_mixture_embedding)

            return separated_audio_embedding


def get(identifier):
    if identifier is None:
        return nn.Identity
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        cls = globals().get(identifier)
        if cls is None:
            raise ValueError("Could not interpret normalization identifier: " + str(identifier))
        return cls
    else:
        raise ValueError("Could not interpret normalization identifier: " + str(identifier))
