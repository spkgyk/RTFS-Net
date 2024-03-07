import sys
import torch
import torch.nn as nn
import pytorch_lightning as ptl

from ..utils import get_MACS_params


class BaseAVModel(nn.Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def load_state_dict_in(model, pretrained_dict):
        model_dict = model.state_dict()
        update_dict = {}
        for k, v in pretrained_dict.items():
            if "audio_model" in k:
                update_dict[k[12:]] = v
        model_dict.update(update_dict)
        model.load_state_dict(model_dict)
        return model

    @staticmethod
    def from_pretrain(pretrained_model_conf_or_path, *args, **kwargs):
        from .. import get

        conf = torch.load(pretrained_model_conf_or_path, map_location="cpu")

        model_class = get(conf["model_name"])
        model = model_class(print_macs=False, *args, **kwargs)
        model.load_state_dict(conf["state_dict"])

        return model

    def serialize(self):
        model_conf = dict(
            model_name=self.__class__.__name__,
            state_dict=self.get_state_dict(),
            model_args=self.get_config(),
        )

        infos = dict()
        infos["software_versions"] = dict(
            torch_version=torch.__version__,
            pytorch_lightning_version=ptl.__version__,
            python_version=sys.version,
        )

        model_conf["infos"] = infos
        return model_conf

    def get_state_dict(self):
        """In case the state dict needs to be modified before sharing the model."""
        return self.state_dict()

    def get_config(self):
        """Should return args to re-instantiate the class."""
        raise NotImplementedError

    def get_MACs(self):
        with torch.no_grad():
            batch_size = 1
            seconds = 2
            device = next(self.parameters()).device

            audio_input = torch.rand(batch_size, seconds * 16000).to(device)

            v_chan = self.pretrained_vout_chan if self.pretrained_vout_chan > 0 else 1

            if self.video_bn_params.get("is2d", False):
                video_input = torch.rand(batch_size, v_chan, seconds * 25, 16).to(device)
            else:
                video_input = torch.rand(batch_size, v_chan, seconds * 25).to(device)

            encoded_audio = self.encoder(audio_input)

            bn_audio = self.audio_bottleneck(encoded_audio)
            bn_video = self.video_bottleneck(video_input)

            separated_audio_embedding = self.mask_generator(bn_audio, encoded_audio)

            MACs = []

            MACs += get_MACS_params(self.encoder, (audio_input,))

            MACs += get_MACS_params(self.audio_bottleneck, (encoded_audio,))

            MACs += get_MACS_params(self.video_bottleneck, (video_input,))

            MACs += get_MACS_params(self.refinement_module, (bn_audio, bn_video))

            MACs += self.refinement_module.get_MACs(bn_audio, bn_video)

            MACs += get_MACS_params(self.mask_generator, (bn_audio, encoded_audio))

            MACs += get_MACS_params(self.decoder, inputs=(separated_audio_embedding, encoded_audio.shape))

            MACs += get_MACS_params(self, inputs=(audio_input, video_input))

            MACs = ["{:,}".format(m) for m in MACs]

            s = (
                "CTCNet\n"
                "Encoder ------------- MACs: {:>8} M    Params: {:>6} K\n"
                "Audio BN ------------ MACs: {:>8} M    Params: {:>6} K\n"
                "Video BN ------------ MACs: {:>8} M    Params: {:>6} K\n"
                "RefinementModule ---- MACs: {:>8} M    Params: {:>6} K\n"
                "   AudioNet --------- MACs: {:>8} M    Params: {:>6} K\n"
                "   VideoNet --------- MACs: {:>8} M    Params: {:>6} K\n"
                "   FusionNet -------- MACs: {:>8} M    Params: {:>6} K\n"
                "Mask Generator ------ MACs: {:>8} M    Params: {:>6} K\n"
                "Decoder ------------- MACs: {:>8} M    Params: {:>6} K\n"
                "Total --------------- MACs: {:>8} M    Params: {:>6} K\n"
            ).format(*MACs)

            self.macs_parms = s
            print(s)
