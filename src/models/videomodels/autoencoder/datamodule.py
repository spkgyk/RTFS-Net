###
# Author: Kai Li
# Date: 2021-06-03 18:29:46
# LastEditors: Please set LastEditors
# LastEditTime: 2022-03-16 06:36:17
###

import os
import json
import torch
import numpy as np
import random as random

from torch.utils.data import Dataset, DataLoader
from .transform import get_preprocessing_pipelines


def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


class AVSpeechDataset(Dataset):
    def __init__(self, json_dir: str = "", segment: float = 4.0, is_train: bool = True):
        super().__init__()
        if json_dir == None:
            raise ValueError("JSON DIR is None!")
        self.json_dir = json_dir
        self.lipreading_preprocessing_func = get_preprocessing_pipelines()["train" if is_train else "val"]
        print("lipreading_preprocessing_func: ", self.lipreading_preprocessing_func)
        if segment is None:
            self.seg_len = None
            self.fps_len = None
        else:
            self.fps_len = int(segment * 25)
        mix_json = os.path.join(json_dir, "mix.json")
        sources_json = [os.path.join(json_dir, source + ".json") for source in ["s1", "s2"]]

        with open(mix_json, "r") as f:
            mix_infos = json.load(f)
        sources_infos = []
        for src_json in sources_json:
            with open(src_json, "r") as f:
                sources_infos.append(json.load(f))

        self.mouths = []

        for i in range(len(mix_infos)):
            for src_inf in sources_infos:
                if src_inf[i][1] not in self.mouths:
                    self.mouths.append(src_inf[i][1])

        print("Total mouths: {}".format(len(self.mouths)))
        self.length = len(self.mouths)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        source_mouth = self.lipreading_preprocessing_func(np.load(self.mouths[idx])["data"])[: self.fps_len]
        return torch.from_numpy(np.array(source_mouth)).float()


class AVSpeechDataModule(object):
    def __init__(
        self,
        train_dir: str,
        valid_dir: str,
        test_dir: str,
        segment: float = 4.0,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ) -> None:
        super().__init__()
        if train_dir == None or valid_dir == None or test_dir == None:
            raise ValueError("JSON DIR is None!")
        # this line allows to access init params with 'self.hparams' attribute
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir
        self.segment = segment
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None

    def setup(self) -> None:
        self.data_train = AVSpeechDataset(json_dir=self.train_dir, segment=self.segment, is_train=True)
        self.data_val = AVSpeechDataset(json_dir=self.valid_dir, segment=self.segment, is_train=False)
        self.data_test = AVSpeechDataset(json_dir=self.test_dir, segment=self.segment, is_train=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    @property
    def make_loader(self):
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()

    @property
    def make_sets(self):
        return self.data_train, self.data_val, self.data_test


if __name__ == "__main__":
    from tqdm import tqdm

    datamodule = AVSpeechDataModule(
        "/home/likai/Autoencoder/code/LRS2/tr",
        "/home/likai/Autoencoder/code/LRS2/cv",
        "/home/likai/Autoencoder/code/LRS2/tt",
        segment=2,
        batch_size=10,
    )
    datamodule.setup()
    train, val, test = datamodule.make_sets
    for idx in tqdm(range(len(train))):
        mouth = train[idx]
        print(mouth.shape)
        import pdb

        pdb.set_trace()
    for idx in tqdm(range(len(val))):
        mouth = train[idx]
    for idx in tqdm(range(len(test))):
        mouth = train[idx]
