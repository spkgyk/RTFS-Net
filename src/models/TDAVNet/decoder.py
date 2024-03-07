import torch
import inspect
import torch.nn as nn


class BaseDecoder(nn.Module):
    def pad_to_input_length(self, separated_audio, input_frames):
        output_frames = separated_audio.shape[-1]
        return nn.functional.pad(separated_audio, [0, input_frames - output_frames])

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def get_config(self):
        encoder_args = {}

        for k, v in (self.__dict__).items():
            if not k.startswith("_") and k != "training":
                if not inspect.ismethod(v):
                    encoder_args[k] = v

        return encoder_args


class ConvolutionalDecoder(BaseDecoder):
    def __init__(
        self,
        in_chan: int,
        n_src: int,
        kernel_size: int,
        stride: int,
        bias=False,
        *args,
        **kwargs,
    ):
        super(ConvolutionalDecoder, self).__init__()

        self.in_chan = in_chan
        self.n_src = n_src
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias

        self.padding = (self.kernel_size - 1) // 2
        self.output_padding = ((self.kernel_size - 1) // 2) - 1

        self.decoder = nn.ConvTranspose1d(
            in_channels=self.in_chan,
            out_channels=1,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            bias=self.bias,
        )

        torch.nn.init.xavier_uniform_(self.decoder.weight)

    def forward(self, separated_audio_embedding: torch.Tensor, input_shape: torch.Size):
        # B, n_src, N, T
        batch_size, length = input_shape[0], input_shape[-1]

        separated_audio_embedding = separated_audio_embedding.view(batch_size * self.n_src, self.in_chan, -1)

        separated_audio = self.decoder(separated_audio_embedding)  # B * n_src, N, T -> B*n_src, 1, L
        separated_audio = self.pad_to_input_length(separated_audio, length)
        separated_audio = separated_audio.view(batch_size, self.n_src, -1)

        return separated_audio


class STFTDecoder(BaseDecoder):
    def __init__(
        self,
        win: int,
        hop_length: int,
        in_chan: int,
        n_src: int,
        kernel_size: int = -1,
        stride: int = 1,
        bias: bool = False,
        *args,
        **kwargs,
    ):
        super(STFTDecoder, self).__init__()
        self.win = win
        self.hop_length = hop_length
        self.in_chan = in_chan
        self.n_src = n_src
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1) // 2
        self.stride = stride
        self.bias = bias

        if self.kernel_size > 0:
            self.decoder = nn.ConvTranspose2d(
                in_channels=self.in_chan,
                out_channels=2,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=self.bias,
            )
            torch.nn.init.xavier_uniform_(self.decoder.weight)
        else:
            self.decoder = nn.Identity()

        self.register_buffer("window", torch.hann_window(self.win), False)

    def forward(self, x: torch.Tensor, input_shape: torch.Size):
        # B, n_src, N, T, F

        batch_size, length = input_shape[0], input_shape[-1]

        x = x.view(batch_size * self.n_src, self.in_chan, *x.shape[-2:])  # B, n_src, N, T, F -> # B * n_src, N, T, F
        decoded_separated_audio = self.decoder(x)  # B * n_src, N, T, F - > B * n_src, 2, T, F

        spec = torch.complex(decoded_separated_audio[:, 0], decoded_separated_audio[:, 1])  # B*n_src, T, F
        # spec = torch.stack([spec.real, spec.imag], dim=-1)  # B*n_src, T, F
        spec = spec.transpose(1, 2).contiguous()  # B*n_src, F, T

        output = torch.istft(
            spec,
            n_fft=self.win,
            hop_length=self.hop_length,
            window=self.window.to(x.device),
            length=length,
        )  # B*n_src, L

        output = output.view(batch_size, self.n_src, length)  # B, n_src, L

        return output


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
