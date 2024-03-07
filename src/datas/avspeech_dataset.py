import os
import json
import torch
import numpy as np
import soundfile as sf

from torch.utils.data import Dataset
from .transform import get_preprocessing_pipelines


def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


class AVSpeechDataset(Dataset):
    def __init__(
        self,
        json_dir: str = "",
        n_src: int = 2,
        sample_rate: int = 8000,
        segment: float = 4.0,
        normalize_audio: bool = False,
        return_src_path: bool = False,
        audio_only: bool = False,
    ):
        super().__init__()
        if json_dir == None:
            raise ValueError("JSON DIR is None!")
        if n_src not in [1, 2]:
            raise ValueError("{} is not in [1, 2]".format(n_src))
        self.json_dir = json_dir
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        self.return_src_path = return_src_path
        self.audio_only = audio_only
        self.lipreading_preprocessing_func = get_preprocessing_pipelines()["train" if segment != None else "val"]
        if segment is None:
            self.seg_len = None
        else:
            self.seg_len = int(segment * sample_rate)
        self.n_src = n_src
        self.test = self.seg_len is None
        mix_json = os.path.join(json_dir, "mix.json")
        sources_json = [os.path.join(json_dir, source + ".json") for source in ["s1", "s2"]]

        with open(mix_json, "r") as f:
            mix_infos = json.load(f)
        sources_infos = []
        for src_json in sources_json:
            with open(src_json, "r") as f:
                sources_infos.append(json.load(f))

        self.mix = []
        self.sources = []
        if self.n_src == 1:
            orig_len = len(mix_infos) * 2
            drop_utt, drop_len = 0, 0
            if not self.test:
                for i in range(len(mix_infos) - 1, -1, -1):
                    if mix_infos[i][1] < self.seg_len:
                        drop_utt = drop_utt + 1
                        drop_len = drop_len + mix_infos[i][1]
                        del mix_infos[i]
                        for src_inf in sources_infos:
                            del src_inf[i]
                    else:
                        for src_inf in sources_infos:
                            self.mix.append(mix_infos[i])
                            self.sources.append(src_inf[i])
            else:
                for i in range(len(mix_infos)):
                    for src_inf in sources_infos:
                        self.mix.append(mix_infos[i])
                        self.sources.append(src_inf[i])

            if drop_utt > 0:
                print(
                    "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                        drop_utt, drop_len / sample_rate / 3600, orig_len, self.seg_len
                    )
                )
            self.length = orig_len

        elif self.n_src == 2:
            orig_len = len(mix_infos)
            drop_utt, drop_len = 0, 0
            if not self.test:
                for i in range(len(mix_infos) - 1, -1, -1):
                    if mix_infos[i][1] < self.seg_len:
                        drop_utt = drop_utt + 1
                        drop_len = drop_len + mix_infos[i][1]
                        del mix_infos[i]
                        for src_inf in sources_infos:
                            del src_inf[i]
                    else:
                        self.mix.append(mix_infos[i])
                        self.sources.append([src_inf[i] for src_inf in sources_infos])
            else:
                self.mix = mix_infos
                self.sources = sources_infos
            if drop_utt > 0:
                print(
                    "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                        drop_utt, drop_len / sample_rate / 3600, orig_len, self.seg_len
                    )
                )
            self.length = orig_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        self.EPS = 1e-8
        if self.n_src == 1:
            rand_start = 0
            stop = self.seg_len

            mix_source, _ = sf.read(self.mix[idx][0], start=rand_start, stop=stop, dtype="float32")
            source = sf.read(self.sources[idx][0], start=rand_start, stop=stop, dtype="float32")[0]
            if not self.audio_only:
                source_mouth = self.lipreading_preprocessing_func(np.load(self.sources[idx][1])["data"])

            source = torch.from_numpy(source)
            mixture = torch.from_numpy(mix_source)

            if self.normalize_audio:
                m_std = mixture.std(-1, keepdim=True)
                mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
                source = normalize_tensor_wav(source, eps=self.EPS, std=m_std)

            common_return = (mixture[: self.sample_rate * 2], source[: self.sample_rate * 2])
            if not self.audio_only:
                common_return += (torch.stack([torch.from_numpy(source_mouth)]),)

            common_return += (self.mix[idx][0].split("/")[-1],)

            if self.return_src_path:
                common_return += (self.sources[idx][0],)

            return common_return

        if self.n_src == 2:
            if self.mix[idx][1] == self.seg_len or self.test:
                rand_start = 0
            else:
                rand_start = np.random.randint(0, self.mix[idx][2] - self.seg_len)

            if self.test:
                stop = None
            else:
                stop = rand_start + self.seg_len
            assert rand_start == 0

            mix_source, _ = sf.read(self.mix[idx][0], start=rand_start, stop=stop, dtype="float32")
            sources = []
            for src in self.sources[idx]:
                # import pdb; pdb.set_trace()
                sources.append(sf.read(src[0], start=rand_start, stop=stop, dtype="float32")[0])

            if not self.audio_only:
                sources_mouths = [
                    torch.from_numpy(self.lipreading_preprocessing_func(np.load(src[1])["data"])) for src in self.sources[idx]
                ]
            sources = torch.stack([torch.from_numpy(source) for source in sources])
            mixture = torch.from_numpy(mix_source)

            if self.normalize_audio:
                m_std = mixture.std(-1, keepdim=True)
                mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
                sources = normalize_tensor_wav(sources, eps=self.EPS, std=m_std)

            return_tuple = (
                mixture[: self.sample_rate * 2],
                sources[: self.sample_rate * 2],
            )

            if not self.audio_only:
                return_tuple += (torch.stack(sources_mouths),)

            return_tuple += (self.mix[idx][0].split("/")[-1],)

            return return_tuple
